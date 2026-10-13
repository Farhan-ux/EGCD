# ============================================================
# Baselines (VCD + SID, universal decoding) -- HuggingFaceTB/SmolVLM2-2.2B-Instruct
#
# Pipeline: AMBER discriminative-relation queries; VCD (noised-image
# amateur) and SID (shallow-layer amateur) applied at every step of
# every query; per-item records with latencies written to ./results/.
#
# Usage: python vcd_smolvlm_instruct.py   (CUDA GPU, fp16)
# ============================================================

import os, json, time, re, datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import LogitsProcessor, LogitsProcessorList
import subprocess

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AMBER_DIR = Path("./AMBER")
IMG_DIR = AMBER_DIR / "image"

# AMBER benchmark: queries + annotations from the official repo, images from
# the official archive. Both are fetched once and reused on subsequent runs.
if not (AMBER_DIR / "data").exists():
    subprocess.run(
        ["git", "clone", "https://github.com/junyangwang0410/AMBER.git", str(AMBER_DIR)],
        check=True,
    )
    print("Cloned AMBER repo")
else:
    print("AMBER repo already exists")

if not IMG_DIR.exists() or len(list(IMG_DIR.glob("AMBER_*.jpg"))) < 1000:
    import gdown
    import zipfile

    print("Downloading AMBER images...")
    url = "https://drive.google.com/uc?id=1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"
    gdown.download(url, "AMBER_images.zip", quiet=False)
    print("Extracting...")
    with zipfile.ZipFile("AMBER_images.zip") as zf:
        zf.extractall(AMBER_DIR)
    for candidate in [AMBER_DIR, AMBER_DIR / "AMBER", AMBER_DIR / "image"]:
        if candidate.is_dir() and len(list(candidate.glob("AMBER_*.jpg"))) > 100:
            IMG_DIR = candidate
            break

jpgs = list(IMG_DIR.glob("AMBER_*.jpg"))
print(f"Found {len(jpgs)} AMBER images in {IMG_DIR}")
assert len(jpgs) >= 1000, f"Expected >= 1000 images, got {len(jpgs)}"

query_rel = json.load(open(AMBER_DIR / "data/query/query_discriminative-relation.json"))
annotations = json.load(open(AMBER_DIR / "data/annotations.json"))

gt_map, subtype_map = {}, {}
for ann in annotations:
    truth = ann.get("truth")
    if isinstance(truth, str):
        gt_map[ann["id"]] = truth.strip().lower()
        subtype_map[ann["id"]] = ann.get("type")

print(f"Relation queries: {len(query_rel)}")

# ============================================================
# MODEL LOADING -- SmolVLM2-2.2B.
# Same Idefics-lineage API. VERIFY with the sanity check before the full run.
# transformers renamed/removed the Vision2Seq auto class across versions, so
# we try the current name first and fall back to the legacy one.
# ============================================================
try:
    from transformers import AutoModelForImageTextToText as _VisionAutoModel
    print("Using AutoModelForImageTextToText")
except ImportError:
    from transformers import AutoModelForVision2Seq as _VisionAutoModel
    print("Using AutoModelForVision2Seq")
from transformers import AutoProcessor

MODEL_TAG = "smolvlm_instruct"
MODEL_NAME = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = _VisionAutoModel.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    device_map={"": 0},
)
model.eval()
TOKENIZER = processor.tokenizer
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================
# Shared helpers: parse rule + the two baseline mechanisms.
# ============================================================
NEG_RE = re.compile(r"\b(not|no|none|isn't|aren't|doesn't|don't|wasn't|weren't|isnt|arent|doesnt|dont|wasnt|werent|n't)\b")

def parse_yes_no(response):
    r = response.strip().lower()
    if r.startswith("yes"): return "yes"
    if r.startswith("no"): return "no"
    for word in re.findall(r"[a-z']+", r):
        if word in ("yes", "no"): return word
    if NEG_RE.search(r): return "no"
    return None


def find_image_key(inputs):
    if "pixel_values" in inputs:
        return "pixel_values"
    for k, v in inputs.items():
        if torch.is_tensor(v) and re.search(r"image|pixel", k, re.I):
            return k
    raise KeyError(f"No image tensor found among keys: {list(inputs.keys())}")


def add_diffusion_noise(image_tensor, noise_step=500):
    # Official VCD utility (DAMO-NLP-SG/VCD, vcd_utils/vcd_add_noise.py):
    # 1000-step sigmoid-beta diffusion schedule, coefficients at `noise_step`.
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = (1 - betas)
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_acp = torch.sqrt(alphas_cumprod)
    sqrt_1macp = torch.sqrt(1 - alphas_cumprod)
    noise = torch.randn_like(image_tensor.float())
    sacp_t = sqrt_acp[noise_step].item()
    s1macp_t = sqrt_1macp[noise_step].item()
    noisy = sacp_t * image_tensor.float() + s1macp_t * noise
    return noisy.to(image_tensor.dtype)


def find_decoder_layers(model):
    # Locate the language-model decoder layers across architectures: the
    # largest ModuleList of attention blocks outside the vision tower.
    cands = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) >= 8:
            if re.search(r"vision|visual|vit|vpm|resampler|clip|patch", name, re.I):
                continue
            first = m[0]
            if any(hasattr(first, a) for a in ("self_attn", "attention", "self_attention", "attn")):
                cands.append((name, m))
    if not cands:
        raise RuntimeError("Could not locate decoder layers.")
    name, layers = max(cands, key=lambda x: len(x[1]))
    print(f"Decoder layers found: {name} ({len(layers)} layers)")
    return layers

# ============================================================
# VCD and SID generation. ALPHA/BETA match our CD+APC (1.0 / 0.1) so the
# comparison isolates the amateur (noised image / shallow layer), not the
# contrast strength. Both are universal: applied at every step of every query.
# ============================================================
PROMPT_SUFFIX = " Answer with only 'Yes' or 'No'."
MAX_NEW_TOKENS = 6
VCD_ALPHA, VCD_BETA, VCD_NOISE_STEP = 1.0, 0.1, 500
SID_LAYER, SID_ALPHA, SID_BETA = 3, 1.0, 0.1
SEED = 0
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SID_CONTROLLER = {"active": False, "captured": None}


class VCDBatchedLogitsProcessor(LogitsProcessor):
    # VCD (Leng et al. 2024), batched two-row formulation:
    #   row 0 = original image, row 1 = Gaussian-noised image.
    # Contrast (1+alpha)*row0 - alpha*row1 with APC on row 0; row 1 is forced
    # to row 0's chosen token so both rows stay token-aligned and row 1's
    # next-step logits are conditional on the same prefix. Everything runs
    # inside ONE standard generate call -- no manual cache handling, which
    # sidesteps the Qwen2-VL mrope and Phi cache quirks of recent transformers.
    def __init__(self, alpha=1.0, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_ids, scores):
        standard = scores[0:1].float()
        noised = scores[1:2].float()
        probs = F.softmax(standard, dim=-1)
        cutoff = self.beta * probs.max(dim=-1, keepdim=True).values
        apc = probs >= cutoff
        vcd = (1 + self.alpha) * standard - self.alpha * noised
        vcd = vcd.masked_fill(~apc, float("-inf"))
        choice = vcd.argmax(dim=-1, keepdim=True)
        forced = torch.full_like(standard, float("-inf"))
        forced.scatter_(1, choice, 0.0)
        return torch.cat([vcd, forced], dim=0)


class SIDShallowController:
    # SID (Huo et al., ICLR 2025): capture the hidden state of a shallow
    # decoder layer at the last position, project it through the LM head, and
    # contrast it with the final logits, bounded by APC.
    def __init__(self, model, layer_idx):
        self.model = model
        self.head = model.get_output_embeddings()
        if self.head is None:
            raise RuntimeError("get_output_embeddings() returned None.")
        self.layers = find_decoder_layers(model)
        self.layer_idx = layer_idx
        self.hook = self.layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not SID_CONTROLLER["active"]:
            return
        h = output[0] if isinstance(output, tuple) else output
        SID_CONTROLLER["captured"] = h[:, -1, :].detach()   # native dtype (fp16); head output is .float()-ed by consumers


class SIDAPCLogitsProcessor(LogitsProcessor):
    def __init__(self, controller, prompt_len, alpha=1.0, beta=0.1):
        self.controller = controller
        self.prompt_len = prompt_len
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_ids, scores):
        captured = SID_CONTROLLER["captured"]
        if captured is None:
            return scores
        head = self.controller.head
        # Device-safe: with device_map="auto" the shallow hidden state
        # (early layers) and the final logits (last device) can live on
        # different GPUs -- route through the head, then align to scores.
        shallow_logits = head(captured.to(head.weight.device)).float().to(scores.device)
        standard_logits = scores.float()
        standard_probs = F.softmax(standard_logits, dim=-1)
        cutoff = self.beta * standard_probs.max(dim=-1, keepdim=True).values
        apc = standard_probs >= cutoff
        out = (1 + self.alpha) * standard_logits - self.alpha * shallow_logits
        return out.masked_fill(~apc, float("-inf"))


sid_controller = SIDShallowController(model, SID_LAYER)


def build_standard_inputs(image, question):
    messages = [{
        "role": "user",
        "content": [{"type": "image"},
                    {"type": "text", "text": question + PROMPT_SUFFIX}],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(text=prompt, images=[image], return_tensors="pt").to(model.device)

def build_cf_prompt_ids(question):
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": question + PROMPT_SUFFIX}],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    return TOKENIZER(prompt, return_tensors="pt").input_ids.to(model.device)

@torch.no_grad()
def generate_vcd(image, question, alpha=VCD_ALPHA, beta=VCD_BETA, max_new_tokens=MAX_NEW_TOKENS):
    t0 = time.perf_counter()
    inputs = build_standard_inputs(image, question)
    img_key = find_image_key(inputs)
    batched = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            nv = add_diffusion_noise(v, VCD_NOISE_STEP) if k == img_key else v
            batched[k] = torch.cat([v, nv], dim=0)
        else:
            batched[k] = v
    proc = LogitsProcessorList([VCDBatchedLogitsProcessor(alpha, beta)])
    out = model.generate(**batched, max_new_tokens=max_new_tokens, do_sample=False,
                         logits_processor=proc)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    gen = out[0][inputs["input_ids"].shape[1]:]   # row 0 = original image
    return processor.decode(gen, skip_special_tokens=True).strip(), len(gen), dt_ms


@torch.no_grad()
def generate_sid(image, question, alpha=SID_ALPHA, beta=SID_BETA, max_new_tokens=MAX_NEW_TOKENS):
    t0 = time.perf_counter()
    inputs = build_standard_inputs(image, question)
    proc = LogitsProcessorList([
        SIDAPCLogitsProcessor(sid_controller, inputs["input_ids"].shape[1], alpha, beta)
    ])
    SID_CONTROLLER["active"] = True
    SID_CONTROLLER["captured"] = None
    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             logits_processor=proc)
    finally:
        SID_CONTROLLER["active"] = False
    dt_ms = (time.perf_counter() - t0) * 1000.0
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip(), len(gen), dt_ms


def both_baselines(image, question):
    v_resp, v_n, v_ms = generate_vcd(image, question)
    s_resp, s_n, s_ms = generate_sid(image, question)
    return {
        "vcd_response": v_resp, "vcd_pred": parse_yes_no(v_resp),
        "sid_response": s_resp, "sid_pred": parse_yes_no(s_resp),
        "vcd_new_tokens": v_n, "sid_new_tokens": s_n,
        "vcd_latency_ms": round(v_ms, 1), "sid_latency_ms": round(s_ms, 1),
    }

# ============================================================
# SANITY CHECK — both baselines on 5 items..
# ============================================================
print(f"Baseline sanity check for {MODEL_TAG} (SID layer {SID_LAYER})...")
for q in query_rel[:5]:
    img_path = IMG_DIR / q["image"]
    if not img_path.exists():
        print(f"  id={q['id']}: image not found, skipping")
        continue
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((1024, 1024))  # VRAM cap
    rec = both_baselines(image, q["query"])
    print(f"  id={q['id']} gt={gt_map.get(q['id'])}"
          f" | VCD: {rec['vcd_response']!r} -> {rec['vcd_pred']} [{rec['vcd_latency_ms']:.0f} ms]"
          f" | SID: {rec['sid_response']!r} -> {rec['sid_pred']} [{rec['sid_latency_ms']:.0f} ms]")

print("\nIf both methods emit parseable yes/no answers, proceed to the full run.")

# ============================================================
# FULL RUN: VCD and SID for every item (universal decoding — no gating).
# Checkpoint every 50 items; safe to stop and re-run the script.
# ============================================================
RUN_START_T = time.time()
RUN_STARTED_UTC = datetime.datetime.utcnow().isoformat() + "Z"

CHECKPOINT_PATH = RESULTS_DIR / f"egcd_baselines_checkpoint_{MODEL_TAG}.json"
results = []
start_idx = 0
if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH) as f:
        results = json.load(f)
    start_idx = len(results)
    print(f"Resuming from sample {start_idx}")

BLANK = {"vcd_response": None, "vcd_pred": None, "sid_response": None, "sid_pred": None,
         "vcd_new_tokens": 0, "sid_new_tokens": 0, "vcd_latency_ms": 0.0, "sid_latency_ms": 0.0}

for i in tqdm(range(start_idx, len(query_rel)), desc=f"Baselines VCD+SID ({MODEL_TAG})"):
    q = query_rel[i]
    img_path = IMG_DIR / q["image"]
    gt = gt_map.get(q["id"])
    subtype = subtype_map.get(q["id"])
    if not img_path.exists() or gt is None:
        continue
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((1024, 1024))  # VRAM cap
    rec = {"id": q["id"], "subtype": subtype, "gt": gt}
    try:
        v_resp, v_n, v_ms = generate_vcd(image, q["query"])
        rec.update({"vcd_response": v_resp, "vcd_pred": parse_yes_no(v_resp),
                    "vcd_new_tokens": v_n, "vcd_latency_ms": round(v_ms, 1)})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    try:
        s_resp, s_n, s_ms = generate_sid(image, q["query"])
        rec.update({"sid_response": s_resp, "sid_pred": parse_yes_no(s_resp),
                    "sid_new_tokens": s_n, "sid_latency_ms": round(s_ms, 1)})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
    for k, v in BLANK.items():
        rec.setdefault(k, v)
    results.append(rec)

    if len(results) % 50 == 0:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(results, f)

with open(CHECKPOINT_PATH, "w") as f:
    json.dump(results, f)

print(f"Done. {len(results)} items processed in {(time.time()-RUN_START_T)/60:.1f} min.")

# ============================================================
# FINAL SAVE.
# ============================================================
meta = {
    "schema": "egcd_baselines_v1",
    "model": MODEL_NAME,
    "model_tag": MODEL_TAG,
    "platform": "local",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "dtype": "float16",
    "max_new_tokens": MAX_NEW_TOKENS,
    "seed": SEED,
    "image_cap": "thumbnail 1024x1024; processor defaults",
    "prompt_suffix": PROMPT_SUFFIX,
    "vcd": {"alpha": VCD_ALPHA, "beta": VCD_BETA, "noise_step": VCD_NOISE_STEP,
            "amateur": "gaussian-noised image (official diffusion schedule)"},
    "sid": {"alpha": SID_ALPHA, "beta": SID_BETA, "shallow_layer_idx": SID_LAYER,
            "amateur": "shallow-layer logits via LM head"},
    "n_items": len(results),
    "run_started_utc": RUN_STARTED_UTC,
    "wall_time_min": round((time.time() - RUN_START_T) / 60.0, 1),
}
final = {**meta, "items": results}

FINAL_NAME = f"egcd_baselines_{MODEL_TAG}.json"
out_path = RESULTS_DIR / FINAL_NAME
with open(out_path, "w") as f:
    json.dump(final, f, indent=2)
print("Saved:", out_path)

# ============================================================
# QUICK LOOK (full set, both universal methods) — final held-out numbers and
# Metrics are computed directly from the per-item records.
# ============================================================
def calc_method(res, key):
    tp = fp = tn = fn = 0
    for r in res:
        pred = r[key]
        gt = r["gt"]
        if r["subtype"] == "discriminative-relation" and gt == "yes":
            if pred == "yes": tp += 1
            else: fn += 1
        elif r["subtype"] == "relation" and gt == "no":
            if pred == "no": tn += 1
            else: fp += 1
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    return recall, spec, (recall + spec) / 2

for key, name in (("vcd_pred", "VCD"), ("sid_pred", "SID")):
    rec, spec, bal = calc_method(results, key)
    print(f"{name:<6} Recall={rec:6.2f}  Specificity={spec:6.2f}  BalancedAcc={bal:6.2f}")

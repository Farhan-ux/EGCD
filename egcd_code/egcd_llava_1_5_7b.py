# ============================================================
# EGCD (Entropy-Gated Contrastive Decoding) -- llava-hf/llava-1.5-7b-hf
#
# Pipeline: AMBER discriminative-relation queries, canonical split
# (seed=42, stratified by image), baseline pass with first-token
# entropy logging, counterfactual CD+APC decoding, tau sweep on the
# tuning half, held-out evaluation, JSON output to ./results/.
#
# Usage: python egcd_llava_1.5_7b.py   (CUDA GPU, fp16)
# ============================================================

import os, json, time, re, datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
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
# CANONICAL SPLIT (identical across all 7 EGCD scripts).
# seed=42, stratified by IMAGE so multi-query images stay in one
# partition. Yields 822 tuning / 842 held-out query pairs.
# ============================================================
SPLIT_SEED = 42
np.random.seed(SPLIT_SEED)

image_to_qids = {}
for q in query_rel:
    image_to_qids.setdefault(q["image"], []).append(q["id"])
image_names = sorted(image_to_qids.keys())
np.random.shuffle(image_names)
n_images = len(image_names)
cut = n_images // 2
tuning_images = set(image_names[:cut])
held_out_images = set(image_names[cut:])

tuning_ids = set()
held_out_ids = set()
for img, qids in image_to_qids.items():
    target = tuning_ids if img in tuning_images else held_out_ids
    for qid in qids:
        target.add(qid)

tuning_queries = [q for q in query_rel if q["id"] in tuning_ids]
held_out_queries = [q for q in query_rel if q["id"] in held_out_ids]
print(f"Tuning  : {len(tuning_queries)} queries across {len(tuning_images)} images")
print(f"Held-out: {len(held_out_queries)} queries across {len(held_out_images)} images")

# ============================================================
# MODEL LOADING -- llava_1.5_7b.
# ============================================================
import transformers
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_TAG = "llava_1.5_7b"
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
TOKENIZER = processor.tokenizer
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"transformers: {transformers.__version__}")

PROMPT_SUFFIX = " Answer with only 'Yes' or 'No'."
MAX_NEW_TOKENS = 6
ALPHA = 1.0
BETA = 0.1
SEED = 0
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

NEG_RE = re.compile(r"\b(not|no|none|isn't|aren't|doesn't|don't|wasn't|weren't|isnt|arent|doesnt|dont|wasnt|werent|n't)\b")

def parse_yes_no(response):
    r = response.strip().lower()
    if r.startswith("yes"): return "yes"
    if r.startswith("no"): return "no"
    for word in re.findall(r"[a-z']+", r):
        if word in ("yes", "no"): return word
    if NEG_RE.search(r): return "no"
    return None


class ContrastiveAPCLogitsProcessor(LogitsProcessor):
    """VCD-style APC contrastive processor.

    amateur = counterfactual forward pass with the IMAGE REMOVED
    (text-only conditional). This is the EGCD System 2 amateur,
    identical across all model variants.
    """
    def __init__(self, model, cf_prompt_ids, prompt_len_standard, alpha=1.0, beta=0.1):
        self.model = model
        self.cf_prompt_ids = cf_prompt_ids
        self.prompt_len_standard = prompt_len_standard
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_ids, scores):
        generated_so_far = input_ids[:, self.prompt_len_standard:]
        cf_input_ids = torch.cat([self.cf_prompt_ids, generated_so_far], dim=1)
        with torch.no_grad():
            cf_out = self.model(input_ids=cf_input_ids, use_cache=False)
        cf_logits = cf_out.logits[:, -1, :].float()

        standard_logits = scores.float()
        standard_probs = F.softmax(standard_logits, dim=-1)
        threshold = self.beta * standard_probs.max(dim=-1, keepdim=True).values
        apc_mask = standard_probs >= threshold
        cd_logits = (1 + self.alpha) * standard_logits - self.alpha * cf_logits
        cd_logits = cd_logits.masked_fill(~apc_mask, float("-inf"))
        return cd_logits

from qwen_vl_utils import process_vision_info

def _apply_chat_template(image, question):
    if image is not None:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": question + PROMPT_SUFFIX}
        ]}]
    else:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": question + PROMPT_SUFFIX}
        ]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def build_standard_inputs(image, question):
    text = _apply_chat_template(image, question)
    batch = processor(text=text, images=image, return_tensors="pt")
    return batch.to(model.device)

def build_cf_prompt_ids(question):
    text = _apply_chat_template(None, question)
    return TOKENIZER(text, return_tensors="pt").input_ids.to(model.device)


@torch.no_grad()
def get_baseline_with_entropy(image, question, max_new_tokens=MAX_NEW_TOKENS):
    inputs = build_standard_inputs(image, question)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True, output_scores=True
    )
    first_token_logits = outputs.scores[0][0]
    probs = F.softmax(first_token_logits.float(), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()

    gen_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(gen_ids, skip_special_tokens=True).strip()
    return response, entropy

@torch.no_grad()
def generate_contrastive(image, question, alpha=ALPHA, beta=BETA, max_new_tokens=MAX_NEW_TOKENS):
    standard_inputs = build_standard_inputs(image, question)
    cf_prompt_ids = build_cf_prompt_ids(question)
    prompt_len_standard = standard_inputs["input_ids"].shape[1]

    processor_list = LogitsProcessorList([
        ContrastiveAPCLogitsProcessor(model, cf_prompt_ids, prompt_len_standard, alpha=alpha, beta=beta)
    ])
    out_ids = model.generate(
        **standard_inputs, max_new_tokens=max_new_tokens, do_sample=False,
        logits_processor=processor_list,
    )
    gen = out_ids[0][prompt_len_standard:]
    return processor.decode(gen, skip_special_tokens=True).strip()

def generate_all_signals_entropy(image, question, alpha=ALPHA, beta=BETA):
    baseline_resp, entropy = get_baseline_with_entropy(image, question)
    cd_resp = generate_contrastive(image, question, alpha, beta)
    return parse_yes_no(baseline_resp), parse_yes_no(cd_resp), entropy

# ============================================================
# SANITY CHECK -- 5 items, no metrics. Verifies the model emits clean
# yes/no and that the entropy value is sensible (nats, > 0).
# ============================================================
print(f"Sanity check for {MODEL_TAG} (5 items)...")
for q in query_rel[:5]:
    img_path = IMG_DIR / q["image"]
    if not img_path.exists():
        print(f"  id={q['id']}: image not found, skipping")
        continue
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((336, 336))  # PIL requires a single tuple arg
    base_resp, cd_resp, ent = generate_all_signals_entropy(image, q["query"])
    print(f"  id={q['id']} gt={gt_map.get(q['id'])} | base={base_resp} cd={cd_resp} H={ent:.4f}")

print("\nIf all five rows show yes/no and a positive entropy, proceed to the full run.")

RUN_START_T = time.time()
RUN_STARTED_UTC = datetime.datetime.utcnow().isoformat() + "Z"

CHECKPOINT_PATH = RESULTS_DIR / f"egcd_checkpoint_{MODEL_TAG}.json"
results = []
start_idx = 0
if CHECKPOINT_PATH.exists():
    with open(CHECKPOINT_PATH) as f:
        results = json.load(f)
    start_idx = len(results)
    print(f"Resuming from sample {start_idx}")

tuning_id_set = set(tuning_ids)
BLANK = {"id": None, "subtype": None, "gt": None, "split": None,
         "baseline_pred": None, "cd_pred": None, "entropy": 0.0,
         "baseline_latency_ms": 0.0, "cd_latency_ms": 0.0}

for i in tqdm(range(start_idx, len(query_rel)), desc=f"EGCD full run ({MODEL_TAG})"):
    q = query_rel[i]
    img_path = IMG_DIR / q["image"]
    gt = gt_map.get(q["id"])
    subtype = subtype_map.get(q["id"])
    if not img_path.exists() or gt is None:
        continue
    image = Image.open(img_path).convert("RGB")
    image.thumbnail((336, 336))  # PIL requires a single tuple arg

    rec = {"id": q["id"], "subtype": subtype, "gt": gt,
            "split": "tuning" if q["id"] in tuning_id_set else "held_out"}
    try:
        t0 = time.perf_counter()
        base_resp, ent_val = get_baseline_with_entropy(image, q["query"])
        rec["baseline_pred"] = parse_yes_no(base_resp)
        rec["baseline_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        rec["entropy"] = float(ent_val)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        rec["baseline_pred"] = None
        rec["entropy"] = 0.0
    try:
        t0 = time.perf_counter()
        cd_resp = generate_contrastive(image, q["query"])
        rec["cd_pred"] = parse_yes_no(cd_resp)
        rec["cd_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        rec["cd_pred"] = None
    for k, v in BLANK.items():
        rec.setdefault(k, v)
    results.append(rec)

    if len(results) % 50 == 0:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(results, f)

with open(CHECKPOINT_PATH, "w") as f:
    json.dump(results, f)
print(f"Done. {len(results)} items in {(time.time()-RUN_START_T)/60:.1f} min.")

def calc_metrics(res, threshold):
    """Return (recall, specificity, balanced_acc, n_evaluated, tp, fn, tn, fp) at threshold τ.

    A query is routed to CD iff its entropy >= threshold; otherwise baseline.
    """
    tp = fp = tn = fn = 0
    n_eval = 0
    for r in res:
        pred = r["cd_pred"] if r["entropy"] >= threshold else r["baseline_pred"]
        if pred is None: continue
        gt = r["gt"]
        if r["subtype"] == "discriminative-relation" and gt == "yes":
            n_eval += 1
            if pred == "yes": tp += 1
            else: fn += 1
        elif r["subtype"] == "relation" and gt == "no":
            n_eval += 1
            if pred == "no": tn += 1
            else: fp += 1
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    spec  = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0
    return recall, spec, (recall + spec) / 2, n_eval, tp, fn, tn, fp

tuning_res = [r for r in results if r["split"] == "tuning"]
held_out_res = [r for r in results if r["split"] == "held_out"]
print(f"Tuning  records: {len(tuning_res)}")
print(f"Held-out records: {len(held_out_res)}")

# Baseline and pure CD+APC on both halves (no threshold gating)
print("\n=== Reference regimes (no gating) ===")
for split_name, res in [("TUNING", tuning_res), ("HELD-OUT", held_out_res)]:
    rec_b, spec_b, bal_b, n_b, *_ = calc_metrics(res, threshold=999.0)
    rec_c, spec_c, bal_c, n_c, *_ = calc_metrics(res, threshold=-1.0)
    print(f"  {split_name:8s} Baseline  R={rec_b:6.2f}  S={spec_b:6.2f}  BalAcc={bal_b:6.2f}  (n={n_b})")
    print(f"  {split_name:8s} CD+APC    R={rec_c:6.2f}  S={spec_c:6.2f}  BalAcc={bal_c:6.2f}  (n={n_c})")

print("\n=== τ sweep on TUNING half (selection) ===")
print(f"{'τ':<8} | {'Recall':<8} | {'Spec':<8} | {'BalAcc':<8} | {'n_eval':<6}")
print("-" * 50)
sweep_data = []
best_tau = None
best_bal = -1.0
for t in np.linspace(0.55, 0.75, 21):
    r, s, b, n, *_ = calc_metrics(tuning_res, float(t))
    sweep_data.append({"threshold": float(t), "recall": r, "specificity": s, "balanced_acc": b, "n_eval": n})
    print(f"{t:<8.3f} | {r:<8.2f} | {s:<8.2f} | {b:<8.2f} | {n:<6}")
    if b > best_bal:
        best_bal = b
        best_tau = float(t)

print(f"\nSelected τ* = {best_tau:.3f}  (tuning-half balanced accuracy = {best_bal:.2f})")

print("\n=== HELD-OUT evaluation at τ* ===")
rec_h, spec_h, bal_h, n_h, tp_h, fn_h, tn_h, fp_h = calc_metrics(held_out_res, best_tau)
rec_b, spec_b, bal_b, n_b, *_ = calc_metrics(held_out_res, 999.0)
rec_c, spec_c, bal_c, n_c, *_ = calc_metrics(held_out_res, -1.0)
print(f"  Held-out Baseline  R={rec_b:6.2f}  S={spec_b:6.2f}  BalAcc={bal_b:6.2f}  (n={n_b})")
print(f"  Held-out CD+APC    R={rec_c:6.2f}  S={spec_c:6.2f}  BalAcc={bal_c:6.2f}  (n={n_c})")
print(f"  Held-out EGCD@τ*   R={rec_h:6.2f}  S={spec_h:6.2f}  BalAcc={bal_h:6.2f}  (n={n_h})  τ*={best_tau:.3f}")
print(f"  Confusion @ τ*: TP={tp_h} FN={fn_h} TN={tn_h} FP={fp_h}")

# ============================================================
# FINAL SAVE -- final_entropy_gated_results_llava_1.5_7b.json
# Schema: per-item records with split tags and latencies, plus held-out
# metrics + confusion counts at τ*.
# ============================================================
meta = {
    "schema": "egcd_split_v1",
    "method": "EGCD (entropy-gated contrastive decoding) with canonical split",
    "model": MODEL_NAME,
    "model_tag": MODEL_TAG,
    "platform": "local",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "dtype": "float16",
    "max_new_tokens": MAX_NEW_TOKENS,
    "alpha": ALPHA,
    "beta": BETA,
    "seed": SEED,
    "split_seed": SPLIT_SEED,
    "split_policy": "stratified_by_image; seed=42; identical across all 7 EGCD scripts",
    "n_tuning": len(tuning_res),
    "n_held_out": len(held_out_res),
    "image_cap": "thumbnail (336, 336)",
    "prompt_suffix": PROMPT_SUFFIX,
    "amateur": "counterfactual forward pass with image omitted (text-only conditional logits)",
    "selected_tau": best_tau,
    "held_out_metrics": {
        "baseline": {"recall": rec_b, "specificity": spec_b, "balanced_acc": bal_b, "n": n_b},
        "cd_apc":   {"recall": rec_c, "specificity": spec_c, "balanced_acc": bal_c, "n": n_c},
        "egcd":     {"recall": rec_h, "specificity": spec_h, "balanced_acc": bal_h, "n": n_h,
                      "tp": tp_h, "fn": fn_h, "tn": tn_h, "fp": fp_h},
    },
    "tuning_sweep": sweep_data,
    "run_started_utc": RUN_STARTED_UTC,
    "wall_time_min": round((time.time() - RUN_START_T) / 60.0, 1),
}

final = {**meta, "raw_results": results}

FINAL_NAME = f"final_entropy_gated_results_{MODEL_TAG}.json"
out_path = RESULTS_DIR / FINAL_NAME
with open(out_path, "w") as f:
    json.dump(final, f, indent=2)
print("Saved:", out_path)

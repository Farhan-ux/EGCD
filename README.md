# EGCD: Entropy-Gated Contrastive Decoding

Code for the paper **"Entropy-Gated Contrastive Decoding"** (EGCD), a
training-free decoding method for large vision-language models that reduces
object hallucination. EGCD contrasts the standard next-token distribution with
a counterfactual amateur distribution (a forward pass conditioned on the
question without the image) under adaptive plausibility constraints, and gates
the contrastive intervention on the first-token predictive entropy of the
baseline pass, so the stronger intervention is applied only where the baseline
is uncertain.

## Repository layout

```
egcd_code/                EGCD method, one standalone script per model
  canonical_split.json    canonical AMBER discriminative-relation split (seed=42)
vcd_code/                 VCD and SID baselines, one standalone script per model
analysis/                 statistical analysis of the per-item outputs
  compute_mcnemar.py      exact two-sided McNemar tests quoted in the paper
requirements.txt
LICENSE
```

Models covered by both the method and the baseline scripts:

- Qwen/Qwen2-VL-2B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- llava-hf/llava-1.5-7b-hf
- llava-hf/llava-v1.6-mistral-7b-hf (LLaVA-NeXT)
- HuggingFaceM4/idefics2-8b
- microsoft/Phi-3.5-vision-instruct
- HuggingFaceTB/SmolVLM2-2.2B-Instruct

## Environment

- Linux with one or more CUDA GPUs; all experiments were run locally on
  NVIDIA hardware in fp16.
- Python 3.10 or later, then:

```
pip install -r requirements.txt
```

## Data

Every script prepares the AMBER benchmark automatically on first run: queries
and annotations are cloned from the official AMBER repository and the image
archive is downloaded and extracted under `./AMBER/image`. Subsequent runs
reuse the local copy, so no manual data steps are required. AMBER
(Wang et al., 2024) is the property of its authors; see their repository for
license and citation details.

## Running

```
python egcd_code/egcd_qwen2_2b.py     # EGCD, one script per model
python vcd_code/vcd_qwen2_2b.py       # VCD + SID baselines, one script per model
```

Each script performs the full pipeline end to end:

1. prepares the AMBER data (first run only),
2. builds the canonical image-stratified split (seed 42; 822 tuning / 842
   held-out query pairs),
3. runs the full query set with checkpointing every 50 items — the run can be
   stopped and resumed,
4. for EGCD scripts: sweeps the entropy threshold tau on the tuning half and
   reports baseline / CD+APC / EGCD metrics on the held-out half,
5. writes per-item JSON records (predictions, entropies, latencies, metrics)
   to `./results/`.

Model weights are pulled from the Hugging Face Hub on first use and cached
locally.

## Statistical analysis

The paired significance tests quoted in the paper are computed by
`analysis/compute_mcnemar.py` (standard library only, no extra
dependencies). The script reads the per-item JSON records written by the
EGCD scripts, re-derives the EGCD answer on every held-out item by applying
the entropy gate at the model's selected tau, and reports exact two-sided
McNemar tests (Dietterich, 1998) of EGCD against universal CD+APC and
against greedy decoding:

```
python analysis/compute_mcnemar.py --results-dir results
python analysis/compute_mcnemar.py --results-dir results --check
```

The first command prints the 2x2 paired outcome tables and writes
`results/mcnemar_results.json`; with `--check` the recomputed discordant-pair
counts and p-values are compared against the values published in the paper
and the script exits non-zero on any mismatch, so it also serves as a
regression test.

## Notes

- `egcd_code/canonical_split.json` records the canonical tuning/held-out
  split shared by all EGCD runs, so every reported number is computed over
  the identical query partition.
- Decoding hyperparameters (alpha = 1.0, beta = 0.1, max_new_tokens = 6,
  seed = 0) are identical across EGCD and the VCD/SID baselines, so the
  comparisons isolate the amateur distribution rather than contrast strength.
- VCD (Leng et al., 2024) uses the official 1000-step diffusion noise
  schedule with noise step 500; SID (Huo et al., 2025) contrasts the
  shallow-layer hidden state (layer 3) projected through the LM head.

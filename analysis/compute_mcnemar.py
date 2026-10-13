#!/usr/bin/env python3
"""Exact two-sided McNemar tests for EGCD vs the decoding baselines.

Reproduces the paired significance statistics quoted in the paper (Section 5.1)
from the per-item records that the egcd_code/egcd_*.py scripts write to
./results/final_entropy_gated_results_<model_tag>.json.

For every model the script

  1. re-derives the EGCD answer on each held-out item by applying the entropy
     gate at the model's selected tau (cd_pred if entropy >= tau else
     baseline_pred), exactly as egcd_code does,
  2. builds the paired 2x2 outcome table for EGCD vs universal CD+APC and
     EGCD vs greedy decoding on the held-out half (N = 842),
  3. computes the exact two-sided McNemar p-value from the discordant pairs
     (two-sided binomial test on min(b, c) of n = b + c; Dietterich, 1998),
  4. cross-checks that the gated answers reproduce the balanced accuracy
     reported in the file's held_out_metrics block,
  5. prints the tables and writes mcnemar_results.json.

With --check the recomputed discordant counts and p-values are compared
against the values published in the paper; the script exits non-zero on any
mismatch, so it doubles as a regression test.

Usage:
    python analysis/compute_mcnemar.py [--results-dir results] [--out PATH]
    python analysis/compute_mcnemar.py --check [--results-dir results]

Standard library only (json / math / argparse / pathlib) - no extra
dependencies. Records with a missing prediction are counted as incorrect,
matching the original computation of the published values.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Published values (paper, Section 5.1): exact two-sided McNemar, EGCD vs
# universal CD+APC on the held-out half. Counts are the discordant pairs:
# egcd_only = EGCD correct & CD+APC wrong, cd_only = CD+APC correct & EGCD
# wrong. For Idefics2-8B the paper additionally quotes EGCD vs greedy decoding.
# ---------------------------------------------------------------------------
PAPER_VALUES = {
    "qwen2_2b":              {"vs_cd_apc": {"egcd_only": 30,  "cd_only": 10,  "p": 0.0022214337732293643}},
    "phi35_vision":          {"vs_cd_apc": {"egcd_only": 12,  "cd_only": 13,  "p": 1.0}},
    "smolvlm_instruct":      {"vs_cd_apc": {"egcd_only": 8,   "cd_only": 8,   "p": 1.0}},
    "qwen2.5_7b":            {"vs_cd_apc": {"egcd_only": 14,  "cd_only": 22,  "p": 0.2429849540349096}},
    "idefics2-8b":           {"vs_cd_apc": {"egcd_only": 7,   "cd_only": 1,   "p": 0.0703125},
                              "vs_greedy":  {"egcd_only": 10,  "cd_only": 11,  "p": 1.0}},
    "llava_1.5_7b":          {"vs_cd_apc": {"egcd_only": 160, "cd_only": 48,  "p": 2.840292018720089e-15}},
    "llava-v1.6-mistral-7b": {"vs_cd_apc": {"egcd_only": 123, "cd_only": 29,  "p": 5.562319435603261e-15}},
}


def norm(pred):
    """Map a raw answer to yes/no the same way the analysis pipeline did."""
    pred = (pred or "").strip().lower()
    if pred.startswith("yes"):
        return "yes"
    if pred.startswith("no"):
        return "no"
    return pred


def exact_mcnemar(b, c):
    """Exact two-sided McNemar p from discordant counts b, c.

    Equivalent to a two-sided exact binomial test on min(b, c) successes out
    of n = b + c trials with p = 0.5 (Dietterich, 1998).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def paired_table(gt, pred_a, pred_b):
    """2x2 paired outcome table: both correct / A-only / B-only / neither."""
    both = a_only = b_only = neither = 0
    for g, a, b in zip(gt, pred_a, pred_b):
        ca, cb = (a == g), (b == g)
        if ca and cb:
            both += 1
        elif ca:
            a_only += 1
        elif cb:
            b_only += 1
        else:
            neither += 1
    return {"both_correct": both, "a_only_correct": a_only,
            "b_only_correct": b_only, "neither_correct": neither,
            "discordant": a_only + b_only, "exact_p": exact_mcnemar(a_only, b_only)}


def balanced_acc(gt, pred):
    tp = sum(1 for g, p in zip(gt, pred) if g == "yes" and p == "yes")
    fn = sum(1 for g, p in zip(gt, pred) if g == "yes" and p != "yes")
    tn = sum(1 for g, p in zip(gt, pred) if g == "no" and p == "no")
    fp = sum(1 for g, p in zip(gt, pred) if g == "no" and p != "no")
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return 100.0 * (rec + spec) / 2.0


def _rename(table):
    """Rename generic a/b keys to the EGCD-centric names used in the paper."""
    return {
        "both_correct": table["both_correct"],
        "egcd_only_correct": table["a_only_correct"],
        "other_only_correct": table["b_only_correct"],
        "neither_correct": table["neither_correct"],
        "discordant": table["discordant"],
        "exact_p": table["exact_p"],
    }


def analyse_file(path):
    """Recompute both McNemar tests for one model's per-item results file."""
    with open(path) as f:
        data = json.load(f)
    tau = data["selected_tau"]
    held = [r for r in data["raw_results"] if r["split"] == "held_out"]
    gt = [r["gt"] for r in held]
    greedy = [norm(r["baseline_pred"]) for r in held]
    cd = [norm(r["cd_pred"]) for r in held]
    egcd = [norm(r["cd_pred"]) if r["entropy"] >= tau else norm(r["baseline_pred"])
            for r in held]

    ba = balanced_acc(gt, egcd)
    gate_check = {
        "egcd_balanced_acc_recomputed": round(ba, 2),
        "egcd_balanced_acc_reported": data["held_out_metrics"]["egcd"]["balanced_acc"],
        "match": abs(ba - data["held_out_metrics"]["egcd"]["balanced_acc"]) < 0.01,
    }

    vs_cd = paired_table(gt, egcd, cd)
    vs_greedy = paired_table(gt, egcd, greedy)
    return {
        "model": data["model"],
        "model_tag": data.get("model_tag", path.stem),
        "selected_tau": tau,
        "n_held_out": len(held),
        "gate_check": gate_check,
        "vs_cd_apc": _rename(vs_cd),
        "vs_greedy": _rename(vs_greedy),
    }


def print_table(tag, res):
    print(f"\n{tag}  ({res['model']}, tau* = {res['selected_tau']:.2f}, "
          f"N = {res['n_held_out']} held-out items)")
    gc = res["gate_check"]
    flag = "OK" if gc["match"] else "MISMATCH"
    print(f"  gate check: EGCD BA recomputed {gc['egcd_balanced_acc_recomputed']:.2f} "
          f"vs reported {gc['egcd_balanced_acc_reported']:.2f}  [{flag}]")
    for label, key in (("CD+APC", "vs_cd_apc"), ("greedy", "vs_greedy")):
        t = res[key]
        print(f"  EGCD vs {label:7s}: both={t['both_correct']:4d}  "
              f"EGCD-only={t['egcd_only_correct']:4d}  "
              f"other-only={t['other_only_correct']:4d}  "
              f"neither={t['neither_correct']:4d}  "
              f"discordant={t['discordant']:4d}  exact p={t['exact_p']:.4g}")


def run_check(results, fail):
    """Compare recomputed values against the published paper values."""
    print("\n== check against published values ==")
    for tag, ref in PAPER_VALUES.items():
        res = results.get(tag)
        if res is None:
            fail(f"{tag}: results file missing")
            continue
        for comp_key, comp_label in (("vs_cd_apc", "vs CD+APC"), ("vs_greedy", "vs greedy")):
            if comp_key not in ref:
                continue
            r = ref[comp_key]
            t = res[comp_key]
            ok = (t["egcd_only_correct"] == r["egcd_only"]
                  and t["other_only_correct"] == r["cd_only"]
                  and math.isclose(t["exact_p"], r["p"], rel_tol=1e-9, abs_tol=1e-15))
            if ok:
                print(f"  [MATCH] {tag:24s} {comp_label:11s} "
                      f"{r['egcd_only']}/{r['cd_only']}  p = {r['p']:.4g}")
            else:
                fail(f"{tag} {comp_label}: recomputed "
                     f"{t['egcd_only_correct']}/{t['other_only_correct']} p={t['exact_p']:.6g} "
                     f"!= published {r['egcd_only']}/{r['cd_only']} p={r['p']:.6g}")
        if not res["gate_check"]["match"]:
            fail(f"{tag}: gated answers do not reproduce the reported held-out "
                 f"balanced accuracy")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", default="results",
                    help="directory holding final_entropy_gated_results_*.json "
                         "(default: ./results)")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: <results-dir>/mcnemar_results.json)")
    ap.add_argument("--check", action="store_true",
                    help="compare recomputed values against the published paper values")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("final_entropy_gated_results_*.json"))
    if not files:
        sys.exit(f"error: no final_entropy_gated_results_*.json files in {results_dir}")

    results, failures = {}, []

    def fail(msg):
        failures.append(msg)
        print(f"  [FAIL] {msg}")

    for path in files:
        res = analyse_file(path)
        results[res["model_tag"]] = res
        print_table(res["model_tag"], res)

    if args.check:
        run_check(results, fail)

    out_path = Path(args.out) if args.out else results_dir / "mcnemar_results.json"
    payload = {
        "schema": "mcnemar-results-v1",
        "test": "exact two-sided McNemar (binomial test on discordant pairs)",
        "reference": "Dietterich (1998)",
        "split": "held_out",
        "gate": "egcd_pred = cd_pred if entropy >= selected_tau else baseline_pred",
        "models": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")

    if failures:
        sys.exit(f"{len(failures)} check failure(s)")


if __name__ == "__main__":
    main()

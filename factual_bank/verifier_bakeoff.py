"""Score a factual bank with the same-model and retrieval-based verifiers.

For each memory of the bank (question, answer, self-grade r, gold grade v) the script records the
verdict of three verifiers. None of them sees the gold answer.

  self_consist   self-consistency by resampling: the model answers the question once more, and the
                 verdict is 1 if the stored answer (its first 40 characters) occurs in the new answer
  adversarial    judge with an adversarial instruction; 1 only if both of two verdicts are 'correct'
  web            retrieval-based verifier: the judge may search the web

`--model` names the model of the first two verifiers and `--retrieval-model` the model of the
retrieval-based verifier. The first two verify with the same model when `--model` is the model that
built the bank.

The verifier bank of a seed holds n memories of the seed's bank: every memory whose self-grade exceeds
its gold grade (b > 0), filled up to n with a seeded random draw from the remaining memories.

Each line of the output trace is one memory:
  idx, topic, r (self-grade), v (gold grade), b = r - v,
  self_consist, adversarial, web       the verdicts
  self_consist_response                the resampled answer
  adversarial_response                 the two replies of the adversarial judge
  web_response                         the reply of the retrieval-based verifier

Usage:
  python factual_bank/verifier_bakeoff.py --seed <seed> --model <model_name>
         --retrieval-model <retrieval_model_name>
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
from factual_bank.call_cache import cached_graders  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "factual_bank")
L = None
MODEL = None
RETRIEVAL_MODEL = None


def load_memories(bank_path: str, n: int, seed: int = 0):
    """The n memories of the verifier bank, in a seeded order; `idx` is the position in that order."""
    rows = []
    with open(bank_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(seed)
    biased = [r for r in rows if r["self_grade"] - r["gold_grade"] > 0]
    other = [r for r in rows if r["self_grade"] - r["gold_grade"] <= 0]
    rng.shuffle(other)
    sel = biased + other[: max(0, n - len(biased))]
    rng.shuffle(sel)
    sel = sel[:n] if n < len(sel) else sel
    for i, m in enumerate(sel):
        m["idx"] = i
    return sel


def arm_self_consist(m):
    """(verdict, the resampled answer)"""
    new = L.answer(m["question"], MODEL)
    ans = m["answer"].strip().lower()[:40]
    return (1.0 if ans in new.strip().lower() else 0.0), new


def arm_adversarial(m):
    """(verdict, the two replies)"""
    replies = [L.verify_adversarial(m["question"], m["answer"], MODEL, vote=i) for i in range(2)]
    return float(all(L.verdict(r) for r in replies)), replies


def arm_web(m):
    """(verdict, the reply)"""
    reply = L.verify_web(m["question"], m["answer"], RETRIEVAL_MODEL)
    return float(L.verdict(reply)), reply


ARMS = {"self_consist": arm_self_consist, "adversarial": arm_adversarial, "web": arm_web}


def _eval_memory(m, arms):
    out = {"idx": m.get("idx"), "topic": m["topic"],
           "r": float(m["self_grade"]), "v": float(m["gold_grade"]),
           "b": float(m["self_grade"] - m["gold_grade"])}
    for name in arms:
        try:
            out[name], out[name + "_response"] = ARMS[name](m)
        except Exception as e:
            out[name] = None
            out[name + "_err"] = str(e)[:120]
    return out


def run(bank: str, n: int, seed: int, out: str, model: str, retrieval_model: str, workers: int = 1,
        arms=None):
    global L, MODEL, RETRIEVAL_MODEL
    L = cached_graders(os.path.join(RESULTS, "cache"))
    MODEL, RETRIEVAL_MODEL = model, retrieval_model
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    arms = arms or list(ARMS.keys())
    mems = load_memories(bank, n, seed)
    nb = sum(1 for m in mems if m["self_grade"] - m["gold_grade"] > 0)
    print(f"Verifier bank of {len(mems)} memories ({nb} with b > 0); verifiers: {', '.join(arms)}; "
          f"model: {model}; retrieval model: {retrieval_model}")

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_eval_memory, m, arms): m for m in mems}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 5 == 0:
                print(f"  {i}/{len(mems)} memories done")

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out)
    failed = sum(1 for r in rows for key in r if key.endswith("_err"))
    if failed:
        print(f"{failed} verifier calls failed (see the *_err fields); run the same command again to repeat them")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=100, help="size of the verifier bank")
    p.add_argument("--model", required=True, help="model of the self-consistency and adversarial verifiers")
    p.add_argument("--retrieval-model", required=True, help="model of the retrieval-based verifier")
    p.add_argument("--bank", default="", help="bank file (default: results/factual_bank/bank_s<seed>.jsonl)")
    p.add_argument("--out", default="", help="output trace (default: results/factual_bank/trace_s<seed>.jsonl)")
    p.add_argument("--workers", type=int, default=1, help="parallel memories (1 = sequential)")
    p.add_argument("--arms", type=str, default="", help="comma-separated verifiers (default: all)")
    a = p.parse_args()
    run(bank=a.bank or os.path.join(RESULTS, "bank_s%d.jsonl" % a.seed), n=a.n, seed=a.seed,
        out=a.out or os.path.join(RESULTS, "trace_s%d.jsonl" % a.seed), model=a.model,
        retrieval_model=a.retrieval_model, workers=a.workers,
        arms=[s.strip() for s in a.arms.split(",") if s.strip()] or None)

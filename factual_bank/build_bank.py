"""Build a self-graded factual memory bank on SimpleQA.

For each sampled question (1) the model answers it, (2) the answer is graded without the gold answer
(the self-grade written to memory), and (3) the answer is graded against the gold answer. Neither the
gold answer nor the third grade is shown to a model that self-grades or verifies the answer. Each of
the three calls has its own model option: `--model` for the answer, `--grader-model` for the
self-grade and `--gold-model` for the gold grade; the same name may be given to several of them. Each
line of the output is one memory:
  question, gold, topic, answer, self_grade, gold_grade, exact_match,
  self_grade_response, gold_grade_response     the replies from which the two grades are computed

The SimpleQA question file is downloaded once to data/simpleqa/. Sampling is a seeded shuffle of the
file followed by taking the first n questions. The model calls of a bank are cached in a directory
next to its output file (<output>_cache), so an interrupted build resumes.

Usage:
  python factual_bank/build_bank.py --seed <seed> --model <model_name>
         --grader-model <grader_model_name> --gold-model <gold_model_name>
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
from factual_bank.call_cache import cached_graders  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "factual_bank")
DATADIR = os.path.join(ROOT, "data", "simpleqa")
SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"


def simpleqa_csv() -> str:
    """Path of the SimpleQA question file; it is downloaded on first use."""
    os.makedirs(DATADIR, exist_ok=True)
    path = os.path.join(DATADIR, "simple_qa_test_set.csv")
    if not os.path.exists(path):
        import requests
        print(f"Downloading SimpleQA -> {path} ...")
        r = requests.get(SIMPLEQA_URL, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def load_simpleqa(n: int, seed: int = 0):
    """n sampled items as (question, gold answer, topic)."""
    rows = []
    with open(simpleqa_csv(), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row.get("problem") or row.get("question")
            gold = row.get("answer")
            topic = "unknown"
            try:
                md = ast.literal_eval(row.get("metadata", "{}"))
                topic = md.get("topic", "unknown")
            except Exception:
                pass
            if q and gold:
                rows.append((q.strip(), gold.strip(), topic))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def leniency_summary(rows) -> dict:
    """Leniency P[self = correct | gold = wrong], aggregate bias, and both per topic."""
    n = len(rows)
    sg = sum(r["self_grade"] for r in rows) / n
    gg = sum(r["gold_grade"] for r in rows) / n
    wrong = [r for r in rows if r["gold_grade"] == 0]
    topics = {}
    for r in rows:
        topics.setdefault(r["topic"], []).append(r)
    by_topic = {}
    for t, sub in sorted(topics.items()):
        sub_wrong = [r for r in sub if r["gold_grade"] == 0]
        by_topic[t] = {
            "n": len(sub), "n_wrong": len(sub_wrong),
            "leniency": round(sum(r["self_grade"] for r in sub_wrong) / len(sub_wrong), 3) if sub_wrong else None,
            "bias": round(sum(r["self_grade"] - r["gold_grade"] for r in sub) / len(sub), 3),
        }
    return dict(n=n, n_wrong=len(wrong), gold_rate=gg, self_rate=sg, aggregate_bias=sg - gg,
                leniency=(sum(r["self_grade"] for r in wrong) / len(wrong)) if wrong else 0.0,
                by_topic=by_topic)


def print_summary(s: dict):
    print("  memories                               : %d (%d wrong)" % (s["n"], s["n_wrong"]))
    print("  answers correct against gold           : %.3f" % s["gold_rate"])
    print("  answers self-graded correct            : %.3f" % s["self_rate"])
    print("  aggregate bias (self - gold)           : %+.3f" % s["aggregate_bias"])
    print("  leniency P[self=correct | gold=wrong]  : %.3f" % s["leniency"])
    print("  by topic:")
    for t, d in s["by_topic"].items():
        if d["leniency"] is not None:
            print("      %-34s n_wrong=%3d  leniency=%.2f  bias=%+.2f" % (t[:34], d["n_wrong"], d["leniency"], d["bias"]))


def run(n: int, seed: int, out: str, model: str, grader_model: str, gold_model: str, workers: int = 1):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # the bank keeps its own call cache: verifier_bakeoff.py asks the same questions again as
    # resamples, which must be new answers and not the ones stored here
    L = cached_graders(os.path.splitext(out)[0] + "_cache")
    qa = load_simpleqa(n, seed)
    print(f"Building a bank of {len(qa)} SimpleQA memories (seed {seed}); answers: {model}; "
          f"self-grade: {grader_model}; gold grade: {gold_model} ...")

    def process(item):
        q, gold, topic = item
        ans = L.answer(q, model)
        sg = L.self_grade(q, ans, grader_model)
        gg = L.gold_grade(q, ans, gold, gold_model)
        exact = gold.lower() in ans.lower()
        return dict(question=q, gold=gold, topic=topic, answer=ans,
                    self_grade=int(L.verdict(sg)), gold_grade=int(L.verdict(gg)), exact_match=int(exact),
                    self_grade_response=sg, gold_grade_response=gg)

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, it): it for it in qa}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"  item failed: {e}")
            if i % 10 == 0:
                print(f"  {i}/{len(qa)} done ({time.time() - t0:.0f}s)")
    if not rows:
        raise SystemExit("No memory was built; see the errors above.")

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out)
    print_summary(leniency_summary(rows))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=150, help="number of questions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", required=True, help="model that answers the questions")
    p.add_argument("--grader-model", required=True, help="model of the self-grade")
    p.add_argument("--gold-model", required=True, help="model of the gold grade")
    p.add_argument("--out", default="", help="output file (default: results/factual_bank/bank_s<seed>.jsonl)")
    p.add_argument("--workers", type=int, default=1, help="parallel questions (1 = sequential)")
    a = p.parse_args()
    run(a.n, a.seed, a.out or os.path.join(RESULTS, "bank_s%d.jsonl" % a.seed), a.model, a.grader_model,
        a.gold_model, workers=a.workers)

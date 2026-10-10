"""Build a self-graded factual bank of a further model.

The model answers each sampled SimpleQA question, and the answer is then graded in a separate call
without the gold answer. The questions are the same seeded sample as in factual_bank/build_bank.py.
The gold answer is not sent to any model: an answer is graded as correct when the gold answer string
occurs in it. `--model` names the model that answers and `--grader-model` the model of the self-grade;
the same name may be given to both.

Each line of the output is one memory:
  idx, question, gold, topic, answer, self_grade, gold_grade, exact_match,
  self_grade_response     the reply from which the self-grade is computed

The model calls of a bank are cached in a directory next to its output file (<output>_cache), so an
interrupted build resumes.

Usage:
  python factual_bank/selfgraded_bank.py --seed <seed> --model <model_name>
         --grader-model <grader_model_name> --name <name>
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)
from factual_bank.build_bank import leniency_summary, load_simpleqa, print_summary  # noqa: E402
from factual_bank.call_cache import cached_graders  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "factual_bank")


def gold_grade(ans, gold):
    """1 if the gold answer string occurs in the answer (case-insensitive), else 0."""
    return 1 if gold.strip().lower() in (ans or "").strip().lower() else 0


def run(n: int, seed: int, out: str, model: str, grader_model: str, workers: int = 1):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    L = cached_graders(os.path.splitext(out)[0] + "_cache")
    qa = load_simpleqa(n, seed)
    print(f"Building a bank of {len(qa)} SimpleQA memories (seed {seed}); answers: {model}; "
          f"self-grade: {grader_model} ...")

    def process(item):
        idx, (q, gold, topic) = item
        ans = L.answer_best_guess(q, model)
        sg = L.self_grade_own(q, ans, grader_model)
        gg = gold_grade(ans, gold)
        return {"idx": idx, "question": q, "gold": gold, "topic": topic, "answer": ans,
                "self_grade": int(L.verdict(sg)), "gold_grade": gg, "exact_match": gg,
                "self_grade_response": sg}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(process, enumerate(qa)))

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d memories -> %s" % (len(rows), out))
    print_summary(leniency_summary(rows))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=150, help="number of questions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", required=True, help="model that answers the questions")
    p.add_argument("--grader-model", required=True, help="model of the self-grade")
    p.add_argument("--name", required=True, help="short name of the bank, used in the file name")
    p.add_argument("--out", default="", help="output file (default: results/factual_bank/<name>_bank_s<seed>.jsonl)")
    p.add_argument("--workers", type=int, default=1, help="parallel questions (1 = sequential)")
    a = p.parse_args()
    run(a.n, a.seed, a.out or os.path.join(RESULTS, "%s_bank_s%d.jsonl" % (a.name, a.seed)), a.model,
        a.grader_model, workers=a.workers)

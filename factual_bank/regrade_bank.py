"""Re-grade the verifier bank with a further model.

The model receives each (question, answer) memory of the verifier bank and decides whether the answer
is correct, without the gold answer. The memories and their order are those of
factual_bank/verifier_bakeoff.load_memories, so `idx` names the same memory as in the verifier trace.

Each line of the output is one memory:
  idx, question, answer, gold, self_grade, gold_grade, topic,
  verdict      1 if the re-grader says 'correct', else 0
  response     the reply from which the verdict is computed

Usage:
  python factual_bank/regrade_bank.py --seed <seed> --model <model_name> --name <name>
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
from factual_bank.call_cache import cached_graders  # noqa: E402
from factual_bank.verifier_bakeoff import load_memories  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "factual_bank")


def run(bank: str, n: int, seed: int, out: str, model: str, workers: int = 1):
    L = cached_graders(os.path.join(RESULTS, "cache"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    mems = load_memories(bank, n, seed)
    print(f"Re-grading {len(mems)} memories of {bank} with {model} ...")

    def process(m):
        reply = L.regrade(str(m.get("question", "")), str(m.get("answer", "")), model)
        return {"idx": m["idx"], "question": m.get("question"), "answer": m.get("answer"),
                "gold": m.get("gold"), "self_grade": m.get("self_grade"),
                "gold_grade": m.get("gold_grade"), "topic": m.get("topic"),
                "verdict": int(L.verdict(reply)), "response": reply}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(process, mems))

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d verdicts -> %s   ('correct' rate %.3f)" % (
        len(rows), out, sum(r["verdict"] for r in rows) / len(rows) if rows else 0.0))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=100, help="size of the verifier bank")
    p.add_argument("--model", required=True, help="model of the re-grader")
    p.add_argument("--name", required=True, help="short name of the re-grader, used in the file name")
    p.add_argument("--bank", default="", help="bank file (default: results/factual_bank/bank_s<seed>.jsonl)")
    p.add_argument("--out", default="", help="output file (default: results/factual_bank/regrade_<name>_s<seed>.jsonl)")
    p.add_argument("--workers", type=int, default=1, help="parallel memories (1 = sequential)")
    a = p.parse_args()
    run(bank=a.bank or os.path.join(RESULTS, "bank_s%d.jsonl" % a.seed), n=a.n, seed=a.seed,
        out=a.out or os.path.join(RESULTS, "regrade_%s_s%d.jsonl" % (a.name, a.seed)), model=a.model,
        workers=a.workers)

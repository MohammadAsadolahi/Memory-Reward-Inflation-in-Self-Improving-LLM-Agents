"""Score-ranked reuse on a factual bank.

The script simulates a memory policy that ranks by stored score on a self-graded bank and counts how
often each memory is reused. Every question of the bank is issued as a query once per epoch. The
candidates of a query are the other memories whose question embedding has a cosine similarity of at
least tau with the query, where tau is a percentile of all pairwise similarities; the top-k candidates
by stored score Q are reused, and each reuse updates Q <- Q + alpha (r - Q) with the memory's
self-grade r and adds one to its reuse count n. The self-grades and gold grades are those measured by
build_bank.py; the question embeddings are computed with the embedding model named by `--embed-model`.

The output trace has one line per memory (idx, topic, r, v, b = r - v, Q, n, question). The script
prints the association between the self-grade bias b and the reuse count n, over all memories and
among the wrong memories (U = 0).

Usage:
  python factual_bank/reuse_loop.py --bank <bank_file> --embed-model <embedding_model_name>
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results", "factual_bank")


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _percentile(vals, pct):
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def reuse_loop(r, embs, epoch_orders, alpha, topk, q_init, tau_pct):
    """Run the loop. `r` are the self-grades, `embs` the question embeddings, and `epoch_orders` one
    permutation of the memory indices per epoch. Returns (Q, n, tau, mean number of candidates)."""
    M = len(r)
    sims = [[_cos(embs[i], embs[j]) for j in range(M)] for i in range(M)]
    offdiag = [sims[i][j] for i in range(M) for j in range(M) if i != j]
    tau = _percentile(offdiag, tau_pct)
    cands_of = [[j for j in range(M) if j != i and sims[i][j] >= tau] for i in range(M)]
    Q = [q_init] * M
    n = [0] * M
    for order in epoch_orders:
        for qi in order:
            cands = list(cands_of[qi])
            if not cands:
                continue
            cands.sort(key=lambda j: Q[j], reverse=True)
            for j in cands[:topk]:
                Q[j] += alpha * (r[j] - Q[j])
                n[j] += 1
    return Q, n, tau, sum(len(c) for c in cands_of) / M


def _cov(xs, ys):
    k = len(xs)
    mx = sum(xs) / k
    my = sum(ys) / k
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / k


def _pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    sx = math.sqrt(_cov(xs, xs)) or 1e-12
    sy = math.sqrt(_cov(ys, ys)) or 1e-12
    return _cov(xs, ys) / (sx * sy)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bank", required=True, help="bank file written by build_bank.py")
    p.add_argument("--embed-model", required=True, help="embedding model of the question embeddings")
    p.add_argument("--out", default=os.path.join(RESULTS, "reuse_loop_trace.jsonl"))
    p.add_argument("--alpha", type=float, default=0.1, help="step of the score update")
    p.add_argument("--topk", type=int, default=2, help="memories reused per query")
    p.add_argument("--epochs", type=int, default=15, help="passes over the query stream")
    p.add_argument("--q_init", type=float, default=0.5, help="initial stored score")
    p.add_argument("--tau_pct", type=float, default=75.0, help="similarity threshold as a percentile")
    p.add_argument("--seed", type=int, default=0, help="seed of the query order")
    a = p.parse_args()

    from factual_bank.call_cache import cached_graders
    L = cached_graders(os.path.join(RESULTS, "cache"))
    mems = [json.loads(l) for l in open(a.bank, encoding="utf-8") if l.strip()]
    M = len(mems)
    r = [float(m["self_grade"]) for m in mems]
    v = [float(m["gold_grade"]) for m in mems]
    b = [ri - vi for ri, vi in zip(r, v)]
    embs = L.embed([m["question"] for m in mems], a.embed_model)

    rng = random.Random(a.seed)
    orders = []
    for _ in range(a.epochs):
        order = list(range(M))
        rng.shuffle(order)
        orders.append(order)
    Q, n, tau, avg_cands = reuse_loop(r, embs, orders, a.alpha, a.topk, a.q_init, a.tau_pct)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in sorted(range(M), key=lambda i: n[i], reverse=True):
            f.write(json.dumps(dict(idx=i, topic=mems[i]["topic"], r=r[i], v=v[i], b=b[i],
                                    Q=round(Q[i], 4), n=n[i], question=mems[i]["question"][:80]),
                               ensure_ascii=False) + "\n")
    print("wrote", a.out)

    ns = [float(x) for x in n]
    wrong = [i for i in range(M) if v[i] == 0]
    print("memories: %d (%d wrong);  tau = %.3f (percentile %.0f);  candidates per query = %.1f"
          % (M, len(wrong), tau, a.tau_pct, avg_cands))
    print("all memories   : Cov(b, n) = %+.3f   Corr(b, n) = %+.3f" % (_cov(b, ns), _pearson(b, ns)))
    bw = [b[i] for i in wrong]; nw = [ns[i] for i in wrong]
    print("wrong memories : Cov(b, n | U=0) = %+.3f   Corr(b, n | U=0) = %+.3f" % (_cov(bw, nw), _pearson(bw, nw)))
    infl = [ns[i] for i in range(M) if b[i] > 0]
    hon = [ns[i] for i in range(M) if b[i] == 0]
    print("mean reuse count: memories with b > 0 = %.2f   memories with b = 0 = %.2f"
          % (sum(infl) / len(infl) if infl else 0.0, sum(hon) / len(hon) if hon else 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

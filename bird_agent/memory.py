"""Episode memory of the BIRD agent.

The memory is append-only and unbounded: every episode is written back. Retrieval returns the top-k
stored episodes by SimCSE cosine similarity between task descriptions. The stored reward of an
episode is attached to it as a label for the planner; it is never used to rank or to filter.

The arms differ only in the reward attached to a retrieved episode:
  memento    the self-grade written with the episode
  lucid      min(self-grade, answer-free score): an episode flagged by the answer-free signal is
             demoted, and no episode is ever promoted
  no-memory  nothing is retrieved
"""
from __future__ import annotations
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import simcse_embedder as SE  # noqa: E402

ARMS = ("no-memory", "memento", "lucid")


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


class EpisodeMemory:
    def __init__(self, arm: str):
        assert arm in ARMS, arm
        self.arm = arm
        self.experiences = []
        self._emb_cache = {}

    def _embed(self, text):
        if text not in self._emb_cache:
            self._emb_cache[text] = SE.embed_one(text)
        return self._emb_cache[text]

    def stored_reward(self, exp: dict) -> float:
        s = float(exp.get("self_score", 0.0))
        if self.arm != "lucid":
            return s
        v = exp.get("answer_free_score", None)
        if v is None:
            return s
        return min(s, float(v))

    def retrieve(self, task_desc: str, db_id: str, k: int = 4) -> list:
        """Top-k episodes by cosine similarity. Each returned item carries `_sim`, `_eff` (the stored
        reward of this arm) and `_same_db` (whether the episode is about the database being queried).
        Only the fields that the planner prompt uses are copied."""
        if self.arm == "no-memory":
            return []
        if not self.experiences:
            return []
        q = self._embed(task_desc)
        scored = []
        for e in self.experiences:
            scored.append((_cos(q, self._embed(e["task_desc"])), e))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for sim, e in scored[:k]:
            out.append({
                "task_id": e["task_id"], "task_desc": e["task_desc"], "code": e["code"],
                "_sim": round(sim, 3),
                "_eff": round(self.stored_reward(e), 3),
                "_same_db": (e.get("db_id") == db_id),
            })
        return out

    def add(self, task_id, task_desc, db_id, plan, sql, self_score, answer_free_score=None):
        self.experiences.append({
            "task_id": task_id, "task_desc": task_desc, "db_id": db_id,
            "plan": plan, "code": sql,
            "self_score": float(self_score),
            "answer_free_score": (None if answer_free_score is None else float(answer_free_score)),
        })

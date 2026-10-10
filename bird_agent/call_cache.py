"""Call cache and episode-log reader for the BIRD runs.

CallCache is a content-addressed, append-only cache for model calls and query executions, keyed by
sha256(operation, payload). A call that has completed is never repeated, so an interrupted run resumes
without repeating work, and arms that issue the same call share one result.

load_done_episodes() reads the episode log of one (arm, seed) run and returns the completed episodes
in order, which is what a resumed run rebuilds its memory from.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading


def _key(op, payload):
    h = hashlib.sha256()
    h.update(op.encode("utf-8")); h.update(b"\x00")
    h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


class CallCache:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._d = {}
        self._lock = threading.Lock()
        self._inflight = {}                    # key -> lock held while that key is being computed
        self._inflight_guard = threading.Lock()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._d[rec["k"]] = rec["v"]
                except Exception:
                    continue                 # an incomplete last line is ignored
        self._fh = open(path, "a", encoding="utf-8")
        self.hits = 0
        self.misses = 0

    def get_or_call(self, op, payload, fn):
        """Return the cached result for (op, payload), or call fn(), store its result and return it.
        Exceptions are not cached. Concurrent callers of the same key compute fn() once and share the
        result; callers of different keys run in parallel."""
        k = _key(op, payload)
        with self._lock:
            if k in self._d:
                self.hits += 1
                return self._d[k]
        with self._inflight_guard:
            klock = self._inflight.setdefault(k, threading.Lock())
        with klock:
            with self._lock:
                if k in self._d:
                    self.hits += 1
                    return self._d[k]
            val = fn()
            with self._lock:
                self._d[k] = val
                self._fh.write(json.dumps({"k": k, "v": val}, ensure_ascii=False) + "\n")
                self._fh.flush()
                self.misses += 1
            return val

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


def load_done_episodes(log_path):
    """Completed episode records of a log, as the contiguous prefix of steps 0..n. A line that does
    not parse or lacks a required field is skipped; if a step occurs twice the first record is kept."""
    if not os.path.exists(log_path):
        return []
    done = []
    for line in open(log_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if all(kf in rec for kf in ("step", "task_id", "gold", "self")):
            done.append(rec)
    by_step = {}
    for r in done:
        by_step.setdefault(r["step"], r)
    contiguous = []
    i = 0
    while i in by_step:
        contiguous.append(by_step[i])
        i += 1
    return contiguous

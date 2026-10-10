"""Call cache for the factual-bank experiments.

CachedLLM wraps factual_bank.graders. A model call is stored under (function name, arguments), which
include the model, and an embedding under (model, text). An interrupted run therefore resumes without
repeating completed calls, and a call that is issued again with the same arguments returns the stored
reply. Set NO_CALL_CACHE=1 to call the model every time.

Every completed call is appended to a JSONL log and flushed at once; a JSON snapshot is rewritten
every few calls. On start the snapshot is read first and the log is replayed on top of it.
"""
from __future__ import annotations
import atexit
import hashlib
import json
import os
import threading

_CACHED = {"answer", "self_grade", "gold_grade", "verify_adversarial", "verify_web", "regrade",
           "answer_best_guess", "self_grade_own"}
_SNAP_EVERY = 25


def _key(name, args, kwargs):
    blob = json.dumps([name, args, sorted(kwargs.items())], default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _embed_key(model, text):
    return json.dumps([model, text], ensure_ascii=False)


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


class CachedLLM:
    def __init__(self, real, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        self._real = real
        self._bypass = bool(os.environ.get("NO_CALL_CACHE"))
        self._lock = threading.Lock()
        self._gpath = os.path.join(cache_dir, "call_cache.json")
        self._epath = os.path.join(cache_dir, "embed_cache.json")
        self._glog_path = os.path.join(cache_dir, "call_cache.log.jsonl")
        self._elog_path = os.path.join(cache_dir, "embed_cache.log.jsonl")
        self._grades = self._load(self._gpath)
        self._embeds = self._load(self._epath)
        self._replay(self._glog_path, self._grades)
        self._replay(self._elog_path, self._embeds)
        self._glog = open(self._glog_path, "a", encoding="utf-8")
        self._elog = open(self._elog_path, "a", encoding="utf-8")
        self._since_snap = 0
        self.hits = 0
        self.misses = 0
        atexit.register(self.close)

    @staticmethod
    def _load(path):
        if os.path.exists(path):
            try:
                return json.load(open(path, encoding="utf-8"))
            except Exception:
                return {}
        return {}

    @staticmethod
    def _replay(path, into):
        if not os.path.exists(path):
            return
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                into[rec["k"]] = rec["val"]
            except Exception:
                continue            # an incomplete last line is ignored

    def _log(self, handle, rec):
        handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        handle.flush()

    def _maybe_snapshot(self):
        self._since_snap += 1
        if self._since_snap >= _SNAP_EVERY:
            _atomic_dump(self._grades, self._gpath)
            _atomic_dump(self._embeds, self._epath)
            self._since_snap = 0

    def close(self):
        with self._lock:
            try:
                _atomic_dump(self._grades, self._gpath)
                _atomic_dump(self._embeds, self._epath)
            except Exception:
                pass
            for h in (getattr(self, "_glog", None), getattr(self, "_elog", None)):
                try:
                    h and h.close()
                except Exception:
                    pass

    def embed(self, texts, model):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        with self._lock:
            miss = [t for t in items if _embed_key(model, t) not in self._embeds]
        if miss:
            vecs = self._real.embed(miss, model)
            with self._lock:
                for t, vec in zip(miss, vecs):
                    self._embeds[_embed_key(model, t)] = vec
                    self._log(self._elog, {"k": _embed_key(model, t), "val": vec})
                self.misses += len(miss)
                self._maybe_snapshot()
        with self._lock:
            self.hits += len(items) - len(miss)
            out = [self._embeds[_embed_key(model, t)] for t in items]
        return out[0] if single else out

    def __getattr__(self, name):
        real_attr = getattr(self._real, name)
        if name not in _CACHED or not callable(real_attr):
            return real_attr

        def wrapped(*args, **kwargs):
            if self._bypass:
                return real_attr(*args, **kwargs)
            k = _key(name, args, kwargs)
            with self._lock:
                if k in self._grades:
                    self.hits += 1
                    return self._grades[k]
            val = real_attr(*args, **kwargs)        # an exception is not cached
            with self._lock:
                self._grades[k] = val
                self._log(self._glog, {"k": k, "name": name, "val": val})
                self.misses += 1
                self._maybe_snapshot()
            return val

        return wrapped


def cached_graders(cache_dir):
    """factual_bank.graders behind the call cache stored in `cache_dir`."""
    from factual_bank import graders
    return CachedLLM(graders, cache_dir)

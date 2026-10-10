"""End-to-end BIRD experiment: no-memory control, Memento-style agent, and LUCID.

One episode of an arm:
  retrieve the top-k most similar past episodes (SimCSE)  ->  planner writes a plan
  ->  executor writes one SQL query  ->  the query is executed in the sandbox
  ->  answer-free signal  ->  self-grade  ->  the episode is written to the arm's memory.

All arms of a run see the same task order, prompts, retrieval depth and scorer, and each arm has its
own memory. The arms differ only in the stored reward that a retrieved episode shows to the planner:
`memento` shows the self-grade, `lucid` shows min(self-grade, answer-free score), and `no-memory`
retrieves nothing. The reference query is executed only to compute execution accuracy (EX); neither
it nor its result is visible to the agent.

Each (arm, seed) run appends one JSON record per episode to
  <results-dir>/<tag>_<arm>_s<seed>.jsonl
with the fields
  step, task_id, arm, seed, db_id, difficulty, task_desc, plan, code,
  self          self-grade (1.0 / 0.0)
  answer_free   score of the answer-free signal (1.0 = not flagged, 0.0 = flagged)
  note          evidence reported by the answer-free signal for a flagged episode (logged only)
  gold          execution accuracy of the episode (None if the reference query could not be executed)
  cand_empty, cand_timeout, ref_error
  retrieved     the retrieved episodes: task_id, eff (stored reward shown), sim, same_db

Model calls and query executions are cached in <results-dir>/<tag>_s<seed>_callcache.jsonl. An
interrupted run is resumed by running the same command again.

The prompt texts of the planner, the executor and the self-grader are placeholders in this repository
(see planner.py); the runner stops until all of them have been replaced.

`--model` names the model of the planner and the executor, and `--grader-model` the model of the
self-grader.

Usage:
  python bird_agent/run_bird.py --seed <seed> --arms no-memory,memento,lucid
         --model <model_name> --grader-model <grader_model_name>
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loader as BL                 # noqa: E402
import planner as BP                # noqa: E402
import executor as BE               # noqa: E402
import self_grader as BG            # noqa: E402
import sql_sandbox as SBX           # noqa: E402
import answer_free as AF            # noqa: E402
import simcse_embedder as SE        # noqa: E402
import call_cache as CC             # noqa: E402
from memory import EpisodeMemory, ARMS as ALL_ARMS   # noqa: E402

RESULTS = os.path.join(_ROOT, "results", "bird")
CALL_CACHE = None
_SCHEMA = {}


# ---------------------------------------------------------------- cached model calls and executions
def make_plan(task_desc, schema, retrieved, model):
    sys_p, user_p = BP.build_plan_prompt(task_desc, schema, retrieved)
    payload = {"sys": sys_p, "user": user_p, "model": model}
    return CALL_CACHE.get_or_call(
        "bird_plan", payload, lambda: BP.plan(task_desc, schema, retrieved, model=model))


def make_sql(task_desc, schema, plan, model):
    sys_p, user_p = BE.build_sql_prompt(task_desc, schema, plan)
    payload = {"sys": sys_p, "user": user_p, "model": model}
    return CALL_CACHE.get_or_call(
        "bird_sql", payload, lambda: BE.execute_plan(task_desc, schema, plan, model=model))


def grade_self(task_desc, schema, sql, grader_model):
    sys_p, user_p = BG.build_self_prompt(task_desc, schema, sql)
    payload = {"sys": sys_p, "user": user_p, "model": grader_model}
    return CALL_CACHE.get_or_call(
        "bird_self", payload, lambda: BG.self_grade(task_desc, schema, sql, model=grader_model))


def schema_ddl(db_id):
    if db_id not in _SCHEMA:
        _SCHEMA[db_id] = BL.schema_ddl(db_id)
    return _SCHEMA[db_id]


def exec_candidate(db_id, db_path, sql, timeout):
    """Candidate execution, cached by (db_id, sql): a step that is recomputed after an interruption
    reads the same rows again, also for a query whose result order is not deterministic."""
    return CALL_CACHE.get_or_call("bird_exec", {"db_id": db_id, "sql": sql},
                                  lambda: SBX.run_candidate(db_path, sql, timeout=timeout))


def exec_reference(db_id, db_path, ref_sql, timeout):
    return CALL_CACHE.get_or_call("bird_ref", {"db_id": db_id, "sql": ref_sql},
                                  lambda: SBX.run_reference(db_path, ref_sql, timeout=max(timeout, 60)))


# ---------------------------------------------------------------- running counters
def _new_ctr():
    return {"n": 0, "gold_sum": 0.0, "gold_n": 0, "self_sum": 0.0, "self_n": 0,
            "af_sum": 0.0, "af_n": 0, "infl_sum": 0.0, "infl_n": 0}


def _update(c, gold, self_, af):
    c["n"] += 1
    if isinstance(self_, (int, float)):
        c["self_sum"] += self_; c["self_n"] += 1
    if isinstance(af, (int, float)):
        c["af_sum"] += af; c["af_n"] += 1
    if isinstance(gold, (int, float)):
        c["gold_sum"] += gold; c["gold_n"] += 1
        if isinstance(self_, (int, float)):
            c["infl_sum"] += (self_ - gold); c["infl_n"] += 1


def _ex(c):
    return c["gold_sum"] / c["gold_n"] if c["gold_n"] else float("nan")


def _mean(s, n):
    return s / n if n else float("nan")


def _print_metrics(state, arms, step, N, t0, final=False):
    el = time.time() - t0
    rate = (step - state["_start0"]) / el if el > 0 and step > state["_start0"] else 0.0
    print("\n" + "=" * 78)
    print("  [%s] step %d/%d   elapsed %.0fs   %.2f steps/s%s" % (
        "final" if final else "progress", step, N, el, rate,
        "" if rate <= 0 else "   eta %.0f min" % ((N - step) / rate / 60.0)))
    for arm in arms:
        c = state[arm]["ctr"]
        print("    %-9s  EX=%.3f  self-grade=%.3f  answer-free=%.3f  self-gold=%+.3f   (scored %d/%d)"
              % (arm, _ex(c), _mean(c["self_sum"], c["self_n"]), _mean(c["af_sum"], c["af_n"]),
                 _mean(c["infl_sum"], c["infl_n"]), c["gold_n"], c["n"]))
    print("=" * 78 + "\n", flush=True)


# ---------------------------------------------------------------- resume
def _config_guard(log_path, config):
    """The configuration of a run is written next to its log on the first run. A resumed run must use
    the same configuration; otherwise the run stops."""
    side = log_path + ".config.json"
    if os.path.exists(side):
        old = json.load(open(side, encoding="utf-8"))
        if old != config:
            diff = {kk: [old.get(kk), config.get(kk)] for kk in set(old) | set(config)
                    if old.get(kk) != config.get(kk)}
            raise RuntimeError("configuration of %s differs from this command: %s\n  -> resume with the "
                               "original arguments, or start a new run with another --tag." % (side, diff))
    else:
        json.dump(config, open(side, "w", encoding="utf-8"), indent=2)


def _resume_arm(arm, log_path, rows_by_id, order, expect_task_ids):
    """Read the completed episodes of this arm, check them against the task order, and rebuild the
    memory and the counters. Returns (memory, next step, counters)."""
    done = CC.load_done_episodes(log_path)
    if len(done) > len(order):
        raise RuntimeError("%s holds %d episodes but the task order has %d" % (log_path, len(done), len(order)))
    for i, rec in enumerate(done):
        if rec["task_id"] != expect_task_ids[i]:
            raise RuntimeError("%s: step %d has task %s but the task order has %s (use another --tag)"
                               % (log_path, i, rec["task_id"], expect_task_ids[i]))
    mem = EpisodeMemory(arm)
    ctr = _new_ctr()
    for rec in done:
        r = rows_by_id.get(rec["task_id"], {})
        mem.add(rec["task_id"], r.get("task_desc", rec.get("task_desc", "")),
                rec.get("db_id", r.get("db_id", "")), rec.get("plan", ""), rec.get("code", ""),
                rec["self"], answer_free_score=rec.get("answer_free"))
        _update(ctr, rec.get("gold"), None if arm == "no-memory" else rec.get("self"), rec.get("answer_free"))
    for e in mem.experiences:
        mem._embed(e["task_desc"])
    return mem, len(done), ctr


# ---------------------------------------------------------------- one episode of one arm
def _step_arm(arm, mem, r, schema, ref, step, seed, model, grader_model, timeout, k, out):
    """Compute one episode. The result is put in out[arm]; the log is written by the caller."""
    try:
        desc, db_id = r["task_desc"], r["db_id"]
        retrieved = mem.retrieve(desc, db_id, k=k)
        plan = make_plan(desc, schema, retrieved, model)
        sql = make_sql(desc, schema, plan, model)
        cand = exec_candidate(db_id, r["db_path"], sql, timeout)
        af_score, af_note = AF.answer_free_signal(desc, sql, cand)
        # the no-memory arm never reads a stored reward, so no self-grade is requested for it
        s_score = 0.0 if arm == "no-memory" else grade_self(desc, schema, sql, grader_model)
        s_metric = None if arm == "no-memory" else s_score
        g_score = SBX.gold_score(cand, ref)
        rec = {"step": step, "task_id": r["task_id"], "arm": arm, "seed": seed, "db_id": db_id,
               "difficulty": r.get("difficulty", ""), "task_desc": desc, "plan": plan, "code": sql,
               "self": s_score, "answer_free": af_score, "gold": g_score, "note": af_note,
               "cand_empty": cand.get("cand_empty"), "cand_timeout": cand.get("cand_timeout"),
               "ref_error": ref.get("ref_error"),
               "retrieved": [{"task_id": e["task_id"], "eff": e["_eff"], "sim": e["_sim"],
                              "same_db": e["_same_db"]} for e in retrieved]}
        mem.add(r["task_id"], desc, db_id, plan, sql, s_score, answer_free_score=af_score)
        out[arm] = ("ok", rec, (g_score, s_metric, af_score))
    except Exception as e:                                   # noqa: BLE001
        out[arm] = ("err", e)


# ---------------------------------------------------------------- driver
def unfilled_prompts():
    """Names of the prompt texts that still hold their <...> placeholder."""
    return ["%s.%s" % (mod.__name__, name) for mod in (BP, BE, BG)
            for name, text in sorted(vars(mod).items())
            if name.isupper() and isinstance(text, str) and re.fullmatch(r"<[^<>]*>", text)]


def build_order(rows, seed):
    """The task order of a seed: one seeded permutation of all questions, shared by all arms."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return order


def run(rows, order, seed, k, model, grader_model, timeout, tag, limit, config, arms, results_dir):
    global CALL_CACHE
    if limit:
        order = order[:limit]
    N = len(order)
    rows_by_id = {r["task_id"]: r for r in rows}
    expect_task_ids = [rows[i]["task_id"] for i in order]
    print("order_sha=%s" % config["order_sha"])

    CALL_CACHE = CC.CallCache(os.path.join(results_dir, "%s_s%d_callcache.jsonl" % (tag, seed)))

    state = {"_start0": 0}
    for arm in arms:
        log = os.path.join(results_dir, "%s_%s_s%d.jsonl" % (tag, arm, seed))
        _config_guard(log, dict(config, arm=arm, seed=seed))
        mem, start, ctr = _resume_arm(arm, log, rows_by_id, order, expect_task_ids)
        state[arm] = {"mem": mem, "start": start, "log": log,
                      "fh": open(log, "a", encoding="utf-8"), "ctr": ctr}
        print("[resume] %-9s %d/%d done  (EX=%.3f)" % (arm, start, N, _ex(ctr)), flush=True)

    start = min(state[a]["start"] for a in arms)
    state["_start0"] = start
    if start >= N:
        print("Nothing to do: all arms are complete (%d/%d)." % (N, N))
        _print_metrics(state, arms, N, N, time.time(), final=True)
        return 0

    SE.embed_one("warmup")
    t0 = time.time()
    for step in range(start, N):
        r = rows[order[step]]
        schema = schema_ddl(r["db_id"])
        ref = exec_reference(r["db_id"], r["db_path"], r["reference_sql"], timeout)

        out, threads = {}, []
        for arm in arms:
            if step >= state[arm]["start"]:
                th = threading.Thread(target=_step_arm,
                                      args=(arm, state[arm]["mem"], r, schema, ref, step, seed, model,
                                            grader_model, timeout, k, out))
                threads.append(th); th.start()
        for th in threads:
            th.join()

        errs = [(arm, v[1]) for arm, v in out.items() if v[0] == "err"]
        if errs:
            # nothing is written for a step in which an arm failed; the step is recomputed on resume
            for arm in arms:
                state[arm]["fh"].close()
            CALL_CACHE.close()
            arm0, e0 = errs[0]
            raise RuntimeError("arm '%s' failed at step %d (%s: %s); run the same command again to resume"
                               % (arm0, step, type(e0).__name__, e0))

        line = "[%5d/%d] %-14s" % (step + 1, N, r["db_id"][:14])
        for arm in arms:
            if arm in out:
                _, rec, (g, s, af) = out[arm]
                state[arm]["fh"].write(json.dumps(rec, ensure_ascii=False) + "\n")
                state[arm]["fh"].flush()
                _update(state[arm]["ctr"], g, s, af)
                line += "  | %-9s g=%s s=%s EX=%.3f" % (
                    arm, ("?" if g is None else "%.0f" % g), ("-" if s is None else "%.0f" % s),
                    _ex(state[arm]["ctr"]))
            else:
                line += "  | %-9s (done)" % arm
        print(line, flush=True)

        if (step + 1) % 100 == 0:
            _print_metrics(state, arms, step + 1, N, t0)

    for arm in arms:
        state[arm]["fh"].close()
    CALL_CACHE.close()
    _print_metrics(state, arms, N, N, t0, final=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="BIRD end-to-end run: no-memory, Memento-style agent, LUCID.")
    ap.add_argument("--seed", type=int, default=0, help="seed of the task permutation")
    ap.add_argument("--arms", default=",".join(ALL_ARMS), help="comma-separated arms to run")
    ap.add_argument("--k", type=int, default=4, help="retrieval depth")
    ap.add_argument("--model", required=True, help="model of the planner and the executor")
    ap.add_argument("--grader-model", required=True, help="model of the self-grader")
    ap.add_argument("--timeout", type=int, default=30, help="timeout of a candidate query in seconds")
    ap.add_argument("--data", default=BL.DATA_JSON, help="BIRD question file (dev.json)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N tasks of the order (0 = all)")
    ap.add_argument("--tag", default="bird", help="prefix of the log and cache files")
    ap.add_argument("--results-dir", default=RESULTS)
    a = ap.parse_args()

    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    for arm in arms:
        if arm not in ALL_ARMS:
            ap.error("unknown arm %r (choose from %s)" % (arm, ", ".join(ALL_ARMS)))

    missing = unfilled_prompts()
    if missing:
        print("The prompt texts of the agent are placeholders in this repository. Replace them before a "
              "run:\n  " + "\n  ".join(missing))
        return 2

    os.makedirs(a.results_dir, exist_ok=True)
    rows = BL.load_bird(a.data)
    order = build_order(rows, a.seed)
    limit = a.limit or len(order)

    # everything that determines the behaviour of a run; --limit is not part of it, because a larger
    # limit only extends the same task order
    full_task_ids = [rows[i]["task_id"] for i in order]
    config = {"k": a.k, "model": a.model, "grader_model": a.grader_model, "timeout": a.timeout,
              "data": os.path.basename(a.data), "n_rows": len(rows),
              "order_sha": hashlib.sha256(json.dumps(full_task_ids).encode()).hexdigest()[:16]}
    print("BIRD: %d tasks (%d databases); seed=%d; arms=%s; k=%d"
          % (len(rows), len({r["db_id"] for r in rows}), a.seed, ",".join(arms), a.k))
    return run(rows, order, a.seed, a.k, a.model, a.grader_model, a.timeout, a.tag, limit, config, arms,
               a.results_dir)


if __name__ == "__main__":
    sys.exit(main())

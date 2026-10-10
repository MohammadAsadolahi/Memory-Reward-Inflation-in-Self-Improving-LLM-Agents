"""SQL execution sandbox.

The candidate query and the reference query run in separate subprocesses with separate timeouts, so
the outcome recorded for the candidate depends only on the candidate.

  run_candidate(db, sql, timeout)  the candidate query, executed twice to detect non-determinism.
                                   A query that fails or times out counts as a wrong answer.
  run_reference(db, sql, timeout)  the reference query, executed once. It is used only for scoring.
                                   If it fails or times out the episode cannot be scored (gold = None).
  gold_score(cand, ref)            execution accuracy (EX) as in the official BIRD evaluation:
                                   1.0 if set(candidate rows) == set(reference rows), else 0.0.

Rows are exchanged as JSON lists. Cells are kept as returned by SQLite, except bytes, which are
replaced by a hexadecimal string.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile

_RUNNER = r'''
import json, sqlite3, sys

DB = sys.argv[1]
spec = json.load(open(sys.argv[2], encoding="utf-8"))
sql = spec["sql"]
runs = int(spec.get("runs", 1))

def norm_cell(c):
    if isinstance(c, bytes):
        return "<blob:%s>" % c.hex()
    return c

def run_once():
    con = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/").replace("%", "%25"), uri=True)
    try:
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return [[norm_cell(c) for c in r] for r in rows], None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:300])
    finally:
        con.close()

out = {}
r1, e1 = run_once()
out["error"] = e1
out["rows"] = r1
out["nondet"] = False
if e1 is None and runs > 1:
    r2, e2 = run_once()
    out["nondet"] = (e2 is not None) or (sorted(map(repr, r1)) != sorted(map(repr, r2)))
print("SQLSBX|" + json.dumps(out), flush=True)
'''


def _run_subprocess(db_path: str, sql: str, runs: int, timeout: int) -> dict:
    """Execute `sql` in a fresh interpreter. Returns {error, rows, nondet, timeout}."""
    if not sql or not sql.strip().rstrip(";").strip():
        return {"error": "EmptySQL: no SQL statement produced", "rows": None,
                "nondet": False, "timeout": False}
    fd_r, runner = tempfile.mkstemp(suffix=".py", prefix="bird_sbx_")
    fd_s, specf = tempfile.mkstemp(suffix=".json", prefix="bird_spec_")
    try:
        with os.fdopen(fd_r, "w", encoding="utf-8") as f:
            f.write(_RUNNER)
        with os.fdopen(fd_s, "w", encoding="utf-8") as f:
            json.dump({"sql": sql, "runs": runs}, f)
        try:
            p = subprocess.run([sys.executable, runner, db_path, specf],
                               capture_output=True, text=True, timeout=timeout,
                               encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return {"error": "Timeout: query exceeded %ds" % timeout, "rows": None,
                    "nondet": False, "timeout": True}
        for line in (p.stdout or "").splitlines():
            if line.startswith("SQLSBX|"):
                out = json.loads(line.split("|", 1)[1])
                return {"error": out.get("error"), "rows": out.get("rows"),
                        "nondet": bool(out.get("nondet")), "timeout": False}
        return {"error": "SandboxError: no output (crash)", "rows": None,
                "nondet": False, "timeout": False}
    finally:
        for pth in (runner, specf):
            try:
                os.remove(pth)
            except OSError:
                pass


def run_candidate(db_path: str, sql: str, timeout: int = 30) -> dict:
    """Execute the candidate query (twice). The answer-free signal reads only this result."""
    r = _run_subprocess(db_path, sql, runs=2, timeout=timeout)
    return {"cand_error": r["error"], "cand_nondet": r["nondet"],
            "cand_rows": r["rows"], "cand_timeout": r["timeout"],
            "cand_empty": (len(r["rows"]) == 0) if r["rows"] is not None else None}


def run_reference(db_path: str, sql: str, timeout: int = 60) -> dict:
    """Execute the reference query (once). Used only to score an episode."""
    r = _run_subprocess(db_path, sql, runs=1, timeout=timeout)
    return {"ref_error": r["error"], "ref_rows": r["rows"]}


def gold_score(cand: dict, ref: dict):
    """Execution accuracy of one episode, or None when the reference query could not be executed."""
    if ref.get("ref_error") is not None or ref.get("ref_rows") is None:
        return None
    if cand.get("cand_error") is not None or cand.get("cand_rows") is None:
        return 0.0
    # set equality of result tuples under Python value equality (1 == 1.0), no rounding
    cset = {tuple(r) for r in cand["cand_rows"]}
    rset = {tuple(r) for r in ref["ref_rows"]}
    return 1.0 if cset == rset else 0.0

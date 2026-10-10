"""BIRD development-set loader.

A row is a dict with the keys
  task_id, db_id, question, evidence, task_desc, reference_sql, difficulty, db_path.
`task_desc` is the question followed by the official evidence string ("HINT: ..."); it is the task
input given to every arm. `schema_ddl()` returns the CREATE TABLE statements of a database, read from
its SQLite file. The same text is shown to the planner, the executor and the self-grader.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DATA_JSON = os.environ.get("BIRD_DEV_JSON", os.path.join(_ROOT, "data", "bird", "dev.json"))
DB_ROOT = os.environ.get("BIRD_DB_ROOT", os.path.join(_ROOT, "data", "bird", "dev_databases"))

_SCHEMA_CACHE = {}


def db_path(db_id: str) -> str:
    return os.path.join(DB_ROOT, db_id, db_id + ".sqlite")


def schema_ddl(db_id: str, max_chars: int = 6000) -> str:
    """CREATE TABLE statements from sqlite_master, whitespace-normalised, ordered by table name and
    capped at `max_chars` characters."""
    if db_id in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db_id]
    # open read-only so that a wrong db_id fails instead of creating an empty database file
    con = sqlite3.connect("file:%s?mode=ro" % db_path(db_id).replace("\\", "/").replace("%", "%25"),
                          uri=True)
    try:
        cur = con.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name")
        parts = []
        for name, ddl in cur.fetchall():
            if not ddl:
                continue
            ddl = re.sub(r"\s+", " ", ddl).strip()
            parts.append(ddl)
    finally:
        con.close()
    text = ";\n".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n-- (schema truncated)"
    _SCHEMA_CACHE[db_id] = text
    return text


def load_bird(path: str = None) -> list:
    """All questions of the file, sorted by question id. A question id that occurs more than once is
    kept once."""
    raw = json.load(open(path or DATA_JSON, encoding="utf-8"))
    rows = []
    seen_ids = set()
    for r in raw:
        if int(r["question_id"]) in seen_ids:
            continue
        seen_ids.add(int(r["question_id"]))
        desc = r["question"].strip()
        ev = (r.get("evidence") or "").strip()
        if ev:
            desc += "\nHINT: " + ev
        rows.append({
            "task_id": int(r["question_id"]),
            "db_id": r["db_id"],
            "question": r["question"].strip(),
            "evidence": ev,
            "task_desc": desc,
            "reference_sql": r["SQL"].strip(),
            "difficulty": r.get("difficulty", ""),
            "db_path": db_path(r["db_id"]),
        })
    rows.sort(key=lambda x: x["task_id"])
    return rows

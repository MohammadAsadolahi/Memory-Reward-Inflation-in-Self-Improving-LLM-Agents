"""Executor of the BIRD agent.

The executor receives the schema, the task description and the plan, and translates the plan into one
SQLite SELECT statement. It does not see the retrieved episodes.

The prompt texts are not included in this repository; the two constants below are placeholders, as in
planner.py. The fields of the user prompt are schema, task and plan. The reply is read by
extract_sql().
"""
from __future__ import annotations
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from common import llm_client as L  # noqa: E402

EXECUTOR_SYS = "<executor system prompt>"
EXECUTOR_USER = "<executor user prompt: %(schema)s %(task)s %(plan)s>"


def extract_sql(raw: str) -> str:
    """The content of the first fenced code block (or the whole reply if there is none), ending with
    one semicolon. The language tag `sql` or `sqlite` after the fence is dropped."""
    m = re.search(r"```[ \t]*(?:sqlite|sql)?[ \t]*\r?\n?(.*?)```", raw or "",
                  re.DOTALL | re.IGNORECASE)
    sql = (m.group(1).strip() if m else (raw or "").strip())
    return sql.rstrip(";").strip() + ";" if sql else ""


def build_sql_prompt(task_desc: str, schema_ddl: str, plan: str) -> tuple:
    user = EXECUTOR_USER % {"schema": schema_ddl.strip(), "task": task_desc.strip(),
                            "plan": (plan or "").strip()}
    return EXECUTOR_SYS, user


def execute_plan(task_desc: str, schema_ddl: str, plan: str, model: str, retries: int = 2) -> str:
    sys_p, user_p = build_sql_prompt(task_desc, schema_ddl, plan)
    return extract_sql(L.chat(sys_p, user_p, model=model, retries=retries))

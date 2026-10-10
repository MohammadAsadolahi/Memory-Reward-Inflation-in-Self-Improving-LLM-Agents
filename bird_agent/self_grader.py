"""Self-grader of the BIRD agent.

The self-grader judges the query of the agent by reading the schema, the task description and the
query. The query is not executed for the grade, and the reference query is not shown. The grade is the
reward written to memory with the episode. The model of the self-grader is named by the caller; it may
be the model of the planner and the executor or another one.

The prompt texts are not included in this repository; the two constants below are placeholders, as in
planner.py. The fields of the user prompt are schema, task and sql. The grade is 1.0 when the first of
the words 'correct' and 'incorrect' in the reply is 'correct', and 0.0 otherwise.
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

SELF_SYS = "<self-grader system prompt>"
SELF_USER = "<self-grader user prompt: %(schema)s %(task)s %(sql)s>"


def _parse(text: str) -> bool:
    t = (text or "").strip().lower()
    m = re.search(r"\b(correct|incorrect)\b", t)
    return bool(m and m.group(1) == "correct")


def build_self_prompt(task_desc: str, schema_ddl: str, sql: str) -> tuple:
    user = SELF_USER % {"schema": schema_ddl.strip(), "task": task_desc.strip(),
                        "sql": (sql or "").strip()}
    return SELF_SYS, user


def self_grade(task_desc: str, schema_ddl: str, sql: str, model: str) -> float:
    sys_p, user_p = build_self_prompt(task_desc, schema_ddl, sql)
    return 1.0 if _parse(L.chat(sys_p, user_p, model=model)) else 0.0

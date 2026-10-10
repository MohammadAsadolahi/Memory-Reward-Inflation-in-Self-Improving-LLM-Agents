"""Planner of the BIRD agent.

The planner receives the database schema, the task description and the retrieved episodes, and writes
a plan for one SQLite SELECT query. A retrieved episode is shown as a positive example when its stored
reward is at least 0.5 and as a negative example otherwise. The prompt is built in the same way for
every arm.

The prompt texts are not included in this repository. Each constant below is a placeholder for one
text and names the fields that the code fills in with the % operator (a literal percent sign in a text
is written %%). Replace every placeholder before a run.
"""
from __future__ import annotations
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# fields of the user prompt: schema = the CREATE TABLE statements, task = the task description,
# cases = the text returned by format_cases()
PLANNER_SYS = "<planner system prompt>"
PLANNER_USER = "<planner user prompt: %(schema)s %(task)s %(cases)s>"
# fields of the retrieved episodes: positive / negative = the examples of a group, one after the other;
# same_db = SAME_DB for an episode about the database being queried, otherwise empty; question = the
# question of the episode without its hint, at most 110 characters; sql = its query on one line, at
# most 300 characters
CASES = "<retrieved episodes: %(positive)s %(negative)s>"
POSITIVE_CASE = "<one positive example: %(same_db)s %(question)s %(sql)s>"
NEGATIVE_CASE = "<one negative example: %(same_db)s %(question)s %(sql)s>"
SAME_DB = "<mark of an episode about the database being queried>"
NO_CASES = "<text shown when no episode is retrieved>"
EMPTY_GROUP = "<text shown for a group without examples>"


def format_cases(retrieved) -> str:
    if not retrieved:
        return NO_CASES
    pos, neg = [], []
    for e in retrieved:
        label_pos = float(e.get("_eff", 0.0)) >= 0.5
        sql = (e.get("code", "") or "").strip().replace("\n", " ")
        if len(sql) > 300:
            sql = sql[:300] + " ..."
        case = {"same_db": SAME_DB if e.get("_same_db") else "",
                "question": (e.get("task_desc", "").split("\nHINT:")[0])[:110], "sql": sql}
        if label_pos:
            pos.append(POSITIVE_CASE % case)
        else:
            neg.append(NEGATIVE_CASE % case)
    return CASES % {"positive": "\n".join(pos) if pos else EMPTY_GROUP,
                    "negative": "\n".join(neg) if neg else EMPTY_GROUP}


def build_plan_prompt(task_desc: str, schema_ddl: str, retrieved) -> tuple:
    user = PLANNER_USER % {"schema": schema_ddl.strip(), "task": task_desc.strip(),
                           "cases": format_cases(retrieved)}
    return PLANNER_SYS, user


def plan(task_desc: str, schema_ddl: str, retrieved, model: str) -> str:
    from common import llm_client as L
    sys_p, user_p = build_plan_prompt(task_desc, schema_ddl, retrieved)
    return L.chat(sys_p, user_p, model=model)

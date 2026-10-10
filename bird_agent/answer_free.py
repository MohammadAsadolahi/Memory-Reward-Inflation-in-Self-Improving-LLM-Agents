"""Answer-free de-inflation signal for text-to-SQL.

The signal reads only the task description, the candidate SQL and the outcome of executing that SQL.
It never reads the reference query or its result, and it makes no model call. Three channels:

  execution          the query fails, times out, or returns different results on two identical runs
  degeneracy         the query runs but returns an empty or all-NULL result
  literal grounding  the query filters on an entity-like string literal that does not occur in the
                     task description

answer_free_signal() returns (score, note). A score of 1.0 means that no channel fired; a score of 0.0
means that the episode is flagged, and `note` says which evidence fired.
"""
from __future__ import annotations
import re

# tokens that are never treated as entity-like literals: SQL keywords and common coded values
_SQL_KW = {"select", "from", "where", "group", "order", "by", "having", "and", "or", "not", "null",
           "asc", "desc", "limit", "as", "on", "join", "inner", "left", "outer", "count", "sum",
           "avg", "min", "max", "distinct", "like", "in", "is", "between", "case", "when", "then",
           "else", "end", "y", "n", "t", "f", "true", "false", "yes", "no", "m", "male", "female"}


def _degenerate(rows) -> bool:
    if rows is None:
        return False
    if len(rows) == 0:
        return True
    return all(all(c is None for c in r) for r in rows)


def _str_literals(sql: str):
    """Quoted string literals of the query, with surrounding blanks and LIKE wildcards removed."""
    lits = re.findall(r"'([^']*)'", sql) + re.findall(r'"([^"]*)"', sql)
    out = []
    for s in lits:
        s2 = s.strip().strip("%").strip()
        if s2:
            out.append(s2)
    return out


def literal_grounding(task_desc: str, sql: str):
    """(fired, note). Fires when a string literal of at least three characters that contains a letter
    and is not in _SQL_KW is absent from the task description, both as a whole and by its first
    token."""
    q = (task_desc or "").lower()
    for lit in _str_literals(sql or ""):
        low = lit.lower()
        if len(low) < 3:
            continue
        if low in _SQL_KW:
            continue
        if not re.search(r"[a-z]", low):
            continue
        head = low.split()[0]
        if low not in q and head not in q:
            return True, "SQL filters on '%s' which is absent from the question" % lit
    return False, None


def answer_free_signal(task_desc: str, sql: str, cand: dict) -> tuple:
    """`cand` is the result of sql_sandbox.run_candidate() for `sql`."""
    # execution
    if cand.get("cand_timeout"):
        return 0.0, "query timed out (runaway join?)"
    if cand.get("cand_error") is not None:
        return 0.0, "execution failed with: %s" % cand["cand_error"]
    if cand.get("cand_nondet"):
        return 0.0, "non-deterministic results across identical runs"
    # degeneracy
    if _degenerate(cand.get("cand_rows")):
        return 0.0, "query ran but returned an empty/all-NULL result for a question that expects an answer"
    # literal grounding
    fired, note = literal_grounding(task_desc, sql)
    if fired:
        return 0.0, note
    return 1.0, None

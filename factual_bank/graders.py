"""Model calls of the factual-bank experiments.

Each function sends one request and returns the reply as text. The model is named by the caller.

  answer(question, model)                   the agent's short answer
  self_grade(question, ans, model)          is the answer correct?  judged without the gold answer
  gold_grade(question, ans, gold, model)    does the answer match the gold answer?  measurement only
  verify_adversarial(question, ans, model)  judge with an adversarial instruction, no gold answer
  verify_web(question, ans, model)          retrieval-based verifier: the judge may search the web,
                                            no gold answer
  regrade(question, ans, model)             re-grade of a stored answer, no gold answer
  answer_best_guess(question, model)        answer prompt of the further-model banks
  self_grade_own(question, ans, model)      self-grade prompt of the further-model banks

A grading reply is one word, 'correct' or 'incorrect'. verdict() maps a reply to True or False, so the
scripts store each reply next to the verdict computed from it.
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.llm_client import chat, chat_web_search, embed  # noqa: E402,F401


def verdict(reply: str) -> bool:
    """True if the reply says 'correct'. The first of the words correct, incorrect, yes, no, true and
    false in the reply decides; a reply without any of them counts as 'incorrect'."""
    m = re.search(r"\b(correct|incorrect|yes|no|true|false)\b", (reply or "").strip().lower())
    return bool(m and m.group(1) in ("correct", "yes", "true"))


def answer(question: str, model: str) -> str:
    sys_p = ("You are answering a factual question. Give ONLY the final answer, as briefly as "
             "possible (a few words or a number). No explanation, no preamble.")
    return chat(sys_p, f"Question: {question}\nAnswer:", model)


_VERDICT_SYS = ("You are a strict grader. Decide whether the candidate answer is correct. "
                "Reply with EXACTLY one word: 'correct' or 'incorrect'. No other text.")


def self_grade(question: str, ans: str, model: str) -> str:
    user = (f"Question: {question}\nCandidate answer: {ans}\n\n"
            "Is the candidate answer correct? Reply 'correct' or 'incorrect'.")
    return chat(_VERDICT_SYS, user, model)


def gold_grade(question: str, ans: str, gold: str, model: str) -> str:
    user = (f"Question: {question}\n[correct_answer]: {gold}\nCandidate answer: {ans}\n\n"
            "Does the candidate answer match the correct answer (allowing trivial formatting "
            "differences and numerical rounding)? Reply 'correct' or 'incorrect'.")
    return chat(_VERDICT_SYS, user, model)


_ADV_SYS = ("You are an ADVERSARIAL fact-checker. Assume the candidate answer is WRONG until proven "
            "right. Look hard for any error. Only say 'correct' if you are confident it is factually "
            "right. Reply with EXACTLY one word: 'correct' or 'incorrect'.")


def verify_adversarial(question: str, ans: str, model: str, vote: int = 0) -> str:
    """`vote` numbers the independent verdicts requested for one memory; it is not sent to the model."""
    user = (f"Question: {question}\nCandidate answer: {ans}\n\n"
            "Is the candidate answer correct? Reply 'correct' or 'incorrect'.")
    return chat(_ADV_SYS, user, model)


def verify_web(question: str, ans: str, model: str) -> str:
    sys_p = ("You are a fact-checker with web access. Look up the facts needed, then decide whether the "
             "candidate answer is correct. Reply with EXACTLY one word: 'correct' or 'incorrect'.")
    user = (f"Question: {question}\nCandidate answer: {ans}\n\n"
            "Verify against authoritative sources. Is the candidate answer correct?")
    return chat_web_search(sys_p, user, model)


# The three prompts below are sent as one user message, without a system message.
def regrade(question: str, ans: str, model: str) -> str:
    return chat("", _VERDICT_SYS + "\n\nQuestion: " + question + "\nCandidate answer: " + ans
                + "\n\nIs the candidate answer correct? Reply 'correct' or 'incorrect'.", model)


_GUESS_SYS = ("Answer the question as accurately as you can. Give ONLY the answer, as briefly as "
              "possible (a name, date, number, or short phrase). If you do not know, give your best guess.")
_OWN_SYS = ("You are grading your OWN previous answer for correctness based only on your own "
            "knowledge (no external lookup). Reply with EXACTLY one word: 'correct' or 'incorrect'. "
            "No other text.")


def answer_best_guess(question: str, model: str) -> str:
    return chat("", _GUESS_SYS + "\n\nQuestion: " + question, model)


def self_grade_own(question: str, ans: str, model: str) -> str:
    return chat("", _OWN_SYS + "\n\nQuestion: " + question + "\nYour answer: " + ans
                + "\n\nIs your answer correct? Reply 'correct' or 'incorrect'.", model)

"""Model client of the experiments: the OpenAI Python client.

  chat(system, user, model)              one chat completion
  chat_web_search(system, user, model)   one reply for which the model may search the web
  embed(texts, model)                    sentence embeddings

The client reads OPENAI_API_KEY from the environment, and OPENAI_BASE_URL when another
OpenAI-compatible endpoint is used; both may be set in a `.env` file in the repository root (see
.env.example). Every model is named by the caller. No sampling parameter is set, so a call runs with
the defaults of the endpoint. To use another provider, replace the three functions of this module.
"""
from __future__ import annotations
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

_client = None


def _get_client() -> OpenAI:
    """The client is created on first use, so that the scripts can be imported without an API key."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _call(request, retries: int):
    last = None
    for a in range(retries):
        try:
            return request()
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"model call failed after {retries} retries: {last}")


def chat(system: str, user: str, model: str, retries: int = 3) -> str:
    """One chat turn. With an empty `system` the request holds the user message only."""
    messages = [{"role": "user", "content": user}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    client = _get_client()
    r = _call(lambda: client.chat.completions.create(model=model, messages=messages), retries)
    return (r.choices[0].message.content or "").strip()


def chat_web_search(system: str, user: str, model: str, retries: int = 3) -> str:
    """One reply for which the model may search the web, through the web-search tool of the Responses
    API. The endpoint and the model must provide this tool."""
    client = _get_client()
    r = _call(lambda: client.responses.create(model=model, instructions=system, input=user,
                                              tools=[{"type": "web_search"}]), retries)
    return (r.output_text or "").strip()


def embed(texts, model: str) -> list:
    if isinstance(texts, str):
        texts = [texts]
    r = _get_client().embeddings.create(model=model, input=texts)
    return [d.embedding for d in r.data]

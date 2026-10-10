"""SimCSE sentence encoder used for memory retrieval.

The configuration follows the retrieval embedding of the released Memento code:

  model       princeton-nlp/sup-simcse-bert-base-uncased (768 dimensions)
  pooling     pooler_output if present, otherwise the [CLS] token
  normalise   L2, so that a dot product is a cosine similarity
  max_length  256 tokens, with truncation; batches of 64
  key         the task description of an episode

The model is loaded on the first call.
"""
from __future__ import annotations
import os
import threading

MODEL = os.environ.get("SIMCSE_MODEL", "princeton-nlp/sup-simcse-bert-base-uncased")
MAX_LEN = int(os.environ.get("SIMCSE_MAX_LEN", "256"))
BATCH = int(os.environ.get("SIMCSE_BATCH", "64"))

_tok = None
_model = None
_device = None
# one shared model: the lock serialises loading and forward passes of concurrent arms
_lock = threading.Lock()


def _ensure_loaded():
    global _tok, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModel
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _tok = AutoTokenizer.from_pretrained(MODEL)
    _model = AutoModel.from_pretrained(MODEL).to(_device)
    _model.eval()


def embed_texts(texts):
    """L2-normalised embeddings as Python lists. Accepts one string or a list of strings."""
    import torch
    import torch.nn.functional as F
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    out = []
    with _lock, torch.no_grad():
        _ensure_loaded()
        for i in range(0, len(items), BATCH):
            batch = items[i:i + BATCH]
            enc = _tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            enc = {k: v.to(_device) for k, v in enc.items()}
            o = _model(**enc)
            if getattr(o, "pooler_output", None) is not None:
                e = o.pooler_output
            else:
                e = o.last_hidden_state[:, 0, :]
            e = F.normalize(e, p=2, dim=1)
            out.extend(e.cpu().tolist())
    return out[0] if single else out


def embed_one(text):
    return embed_texts([text])[0]

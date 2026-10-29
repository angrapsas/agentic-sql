"""Backends: how we call a model. Separate from setups, which decide what we ask it.

Keeping these apart is deliberate. Last phase the fine-tuned model could only be
served through transformers while the agent ran on Ollama, and the two got
compared as if the only difference were the fine-tune. A backend is the serving
path; a setup is the scaffold. Changing one should never silently change the other.

Every backend exposes:

    generate(prompt: str) -> (text: str, tokens_in: int, tokens_out: int)
    name: str
    supports_tools: bool     # True if it can do structured tool calls
"""

REGISTRY = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def get_backend(name, **kw):
    if name not in REGISTRY:
        raise SystemExit(f"unknown backend {name!r}. known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)


def available():
    return sorted(REGISTRY)


# importing these registers them; each imports its heavy dependencies lazily
from . import mock, hf, ollama, anthropic  # noqa: E402,F401

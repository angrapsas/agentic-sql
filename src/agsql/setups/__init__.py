"""Setups: what we ask the model to do. Separate from backends, which decide how
we call it.

Every setup is built by a factory taking (backend, memory, **kw) and returning a
callable:

    arm(question: str) -> dict with 'sql' plus whatever metrics it can report
"""

REGISTRY = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def get_setup(name, **kw):
    if name not in REGISTRY:
        raise SystemExit(f"unknown setup {name!r}. known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)


def available():
    return sorted(REGISTRY)


from . import templates, oneshot, loop  # noqa: E402,F401

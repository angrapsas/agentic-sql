"""A scripted backend, for testing the plumbing without a GPU or an API key.

Used by scripts/smoke.sh and the tests. It is the only way to check the setups,
grading and reporting end to end without paying for inference.
"""

from . import register


class MockBackend:
    name = "mock"
    supports_tools = False

    def __init__(self, script=None, thinking=False):
        # script: list of replies to cycle through. Default plays a sensible
        # schema-peek-then-query trajectory.
        self.script = script or [
            "PRAGMA table_info(Track)",
            "SELECT COUNT(*) FROM Track",
            "ANSWER: 3503",
        ]
        self.i = 0
        self.thinking = thinking

    def generate(self, prompt):
        text = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return text, len(prompt) // 4, max(1, len(text) // 4)

    def reset(self):
        self.i = 0


@register("mock")
def _make(**kw):
    return MockBackend(**kw)

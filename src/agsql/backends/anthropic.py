"""Anthropic API backend, for the frontier-model ceiling arm.

The model string is not hardcoded: pass --model-id, or set AGSQL_ANTHROPIC_MODEL.
Model identifiers change, and a stale default silently measures the wrong model.
"""

import os

from . import register


class AnthropicBackend:
    supports_tools = True

    def __init__(self, model_id=None, thinking=False, max_tokens=1024,
                 thinking_budget=2048):
        import anthropic

        self.model = model_id or os.environ.get("AGSQL_ANTHROPIC_MODEL")
        if not self.model:
            raise SystemExit(
                "no Anthropic model given. Pass --model-id <exact model string> "
                "or set AGSQL_ANTHROPIC_MODEL. Check the current identifiers in "
                "the Anthropic docs rather than guessing.")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")

        self.client = anthropic.Anthropic()
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.name = f"anthropic:{self.model}"
        self.cost_tokens = [0, 0]   # running totals, for cost per question

    def generate(self, prompt):
        kw = dict(model=self.model, max_tokens=self.max_tokens,
                  messages=[{"role": "user", "content": prompt}])
        if self.thinking:
            # extended thinking needs max_tokens above the thinking budget
            kw["max_tokens"] = max(self.max_tokens, self.thinking_budget + 512)
            kw["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget}

        r = self.client.messages.create(**kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        tin, tout = r.usage.input_tokens, r.usage.output_tokens
        self.cost_tokens[0] += tin
        self.cost_tokens[1] += tout
        return text, tin, tout


@register("anthropic")
def _make(**kw):
    return AnthropicBackend(**kw)

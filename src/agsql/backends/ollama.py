"""Ollama backend. Cannot serve a LoRA adapter (ADAPTER was removed from
`ollama create` in v0.34.1, and GGUF conversion currently fails for Qwen3.5),
so the tuned arms cannot run here. Kept because the phase 2 agent used it and
it is the bridge to those results.
"""

from . import register


class OllamaBackend:
    supports_tools = True

    def __init__(self, model, thinking=False, host=None):
        import ollama
        self.client = ollama.Client(host=host) if host else ollama
        self.model = model
        self.thinking = thinking
        self.name = f"ollama:{model}"

    def generate(self, prompt):
        r = self.client.chat(model=self.model, think=self.thinking,
                             messages=[{"role": "user", "content": prompt}],
                             options={"temperature": 0, "seed": 0,
                                      "top_p": 1, "top_k": 0,
                                      "repeat_penalty": 1, "presence_penalty": 0})
        text = r["message"]["content"]
        return (text, r.get("prompt_eval_count", 0) or 0, r.get("eval_count", 0) or 0)


@register("ollama")
def _make(**kw):
    return OllamaBackend(**kw)

"""Transformers backend, with an optional PEFT adapter.

This is the only path that can serve a LoRA adapter, which is why the tuned arms
run here. Pass adapter=None for the untuned arm so both run through identical
code and the only difference is the weights.
"""

import os

from . import register


class HFBackend:
    supports_tools = False

    def __init__(self, model_id, adapter=None, thinking=False, quantize=False,
                 max_new_tokens=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.thinking = thinking
        # thinking needs room: reasoning comes out of the same budget as the SQL,
        # and truncating mid-reasoning looks exactly like "the model wrote no SQL"
        self.max_new_tokens = max_new_tokens or (400 if thinking else 150)
        self.name = f"hf:{model_id}" + (f"+{os.path.basename(adapter)}" if adapter else "")

        if dtype is None:
            # T4 is pre-Ampere: no bf16. Ampere and later prefer it.
            ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if ok else torch.float16

        kw = dict(device_map="auto", torch_dtype=dtype)
        if quantize:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype)

        self.tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model
        self.torch = torch

    def generate(self, prompt):
        p = self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=self.thinking)
        ids = self.tok(p, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens,
                                      do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        new = out[0][ids["input_ids"].shape[1]:]
        return (self.tok.decode(new, skip_special_tokens=True),
                int(ids["input_ids"].shape[1]), int(len(new)))


@register("hf")
def _make(**kw):
    return HFBackend(**kw)

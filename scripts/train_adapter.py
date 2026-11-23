#!/usr/bin/env python3
"""Train a LoRA adapter on the question/SQL pairs.

The prompt format is imported from the oneshot setup rather than written out
again here, so that training and evaluation cannot drift apart. The adapter sees
during training exactly the format it will see when it is measured.

Loss is computed on the SQL only. The prompt tokens are masked out, since there
is nothing to learn from predicting the question back.

    python scripts/make_training_data.py
    python scripts/train_adapter.py --out /content/drive/MyDrive/sql_checker/sql_adapter_2b

On a Colab T4, a 2b model trains in fp16 without quantization. Pass --quantize
for a 4b or larger model, which needs 4-bit to fit.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "train_v1.jsonl"))
    ap.add_argument("--out", required=True, help="where to save the adapter")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--quantize", action="store_true",
                    help="4-bit. Needed for 4b and up on a 16GB card.")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="hold out this fraction of pairs: logs validation loss "
                         "every epoch and scores execution accuracy on them after")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from torch.utils.data import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model

    from agsql.setups.oneshot import PROMPT, table_names

    pairs = [json.loads(l) for l in open(args.train)]
    if not pairs:
        raise SystemExit(f"no training pairs in {args.train}")
    tables = ", ".join(table_names())
    train_pairs, val_pairs = split_pairs(pairs, args.val_frac, args.seed)
    print(f"{len(train_pairs)} training pairs, {len(val_pairs)} held out "
          f"| model {args.model_id} | epochs {args.epochs}")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # A T4 is pre-Ampere and has no bf16. Ampere and later prefer it.
    ampere = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if ampere else torch.float16

    load_kw = dict(device_map="auto", torch_dtype=dtype)
    if args.quantize:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype)

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kw)
    model.config.use_cache = False

    # PEFT cannot infer target_modules for this architecture, so they are listed.
    lora = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    print_lora_coverage(model)

    class SQLPairs(Dataset):
        """Prompt tokens are masked so the loss only covers the SQL."""

        def __init__(self, pairs):
            self.rows = []
            for p in pairs:
                prompt = tok.apply_chat_template(
                    [{"role": "user",
                      "content": PROMPT.format(tables=tables, question=p["question"])}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                target = p["sql"] + tok.eos_token
                pi = tok(prompt, add_special_tokens=False)["input_ids"]
                ti = tok(target, add_special_tokens=False)["input_ids"]
                ids = (pi + ti)[:args.max_len]
                labels = ([-100] * len(pi) + ti)[:args.max_len]
                self.rows.append({"input_ids": ids, "labels": labels})

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        return {
            "input_ids": torch.tensor(
                [b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"]))
                 for b in batch]),
            "labels": torch.tensor(
                [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        }

    ds = SQLPairs(train_pairs)
    val_ds = SQLPairs(val_pairs) if val_pairs else None
    print(f"tokenised, median length {sorted(len(r['input_ids']) for r in ds.rows)[len(ds)//2]}")

    targs = TrainingArguments(
        output_dir=os.path.join(ROOT, "results", "_train"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=not ampere,
        bf16=ampere,
        optim="adamw_torch",
        seed=args.seed,
        eval_strategy="epoch" if val_ds else "no",
        per_device_eval_batch_size=args.batch,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      eval_dataset=val_ds, data_collator=collate)

    # With fp16 the GradScaler needs fp32 parameters to scale, and PEFT leaves
    # the adapter weights in the model's dtype. Without this the first backward
    # pass fails. Harmless on bf16.
    if not ampere:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # Loss says nothing about whether the SQL is right. Execute it.
    model.config.use_cache = True
    model.eval()
    report = {"model_id": args.model_id, "epochs": args.epochs,
              "n_train": len(train_pairs), "n_val": len(val_pairs),
              "seed": args.seed, "log": trainer.state.log_history}
    for name, subset in (("train", train_pairs[:40]), ("val", val_pairs)):
        if not subset:
            continue
        ok = sum(execution_match(generate_sql(model, tok, PROMPT, tables,
                                              p["question"], torch), p["sql"])
                 for p in subset)
        report[f"{name}_exec_acc"] = ok / len(subset)
        print(f"{name} execution accuracy: {ok}/{len(subset)}")
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\nadapter saved to {args.out}")
    print("point the grid's adapter path at it, then:")
    print("  python -m agsql.cli run --arm qwen2b-loop-tuned --smoke")


def split_pairs(pairs, frac, seed):
    """Deterministic held-out split. Same seed and file give the same split, so
    adapters trained for different epochs are scored on the same questions."""
    if frac <= 0:
        return list(pairs), []
    import random
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, round(len(pairs) * frac))
    val = set(idx[:n_val])
    return ([p for i, p in enumerate(pairs) if i not in val],
            [p for i, p in enumerate(pairs) if i in val])


def print_lora_coverage(model):
    """Which projections got an adapter, and in how many layers. Qwen3.5 mixes
    full-attention and linear-attention layers with different module names, so
    target_modules can silently miss most of the attention."""
    import re
    from collections import Counter
    hits = Counter()
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and len(getattr(mod, "lora_A", {})):
            hits[re.sub(r"\.\d+\.", ".N.", name).split("base_model.model.")[-1]] += 1
    print("LoRA coverage (module: layers):")
    for k, v in sorted(hits.items()):
        print(f"  {k}: {v}")


def generate_sql(model, tok, prompt_tmpl, tables, question, torch):
    from agsql.grading import extract_sql
    p = tok.apply_chat_template(
        [{"role": "user", "content": prompt_tmpl.format(tables=tables, question=question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=150, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_sql(text)


def execution_match(sql, gold):
    """Same rows as the gold query, order ignored. Errors count as wrong."""
    import sqlite3
    from agsql.dataset import DB_PATH
    con = sqlite3.connect(DB_PATH)
    try:
        got = sorted(map(repr, con.execute(sql).fetchall()))
        want = sorted(map(repr, con.execute(gold).fetchall()))
        return got == want
    except Exception:
        return False
    finally:
        con.close()


if __name__ == "__main__":
    main()

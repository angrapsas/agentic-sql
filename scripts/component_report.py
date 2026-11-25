#!/usr/bin/env python3
"""Score each pipeline component on its own, from the traces saved with results.

For every pipeline result file:
  end to end   strict and loose accuracy, overall and split into in-scope
               (questions the coded generator has a template for) and out-of-scope
  router       accuracy against the gold answerable flag, and how often it
               wrongly refuses an answerable question or lets an unanswerable through
  linker       recall and precision of the linked tables against the tables the
               gold SQL uses (answerable questions the router passed)
  repair       how often it was needed, and how often it produced a query that runs
  cost         model calls, tokens and seconds per question

    python scripts/component_report.py results/pipeline/*.json
Set AGSQL_DB for the renamed schema, as for every other run.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from agsql.setups.pipeline import Schema  # noqa: E402
from agsql.setups.templates import COMPILED  # noqa: E402


def gold_tables(sql, tables):
    return {t for t in tables if re.search(r"\b%s\b" % re.escape(t), sql or "")}


def in_scope(question):
    return any(p.search(question.strip()) for p, _ in COMPILED)


def pct(a, b):
    return f"{100 * a / b:5.1f}" if b else "  n/a"


def main(paths, dataset_name):
    from agsql import dataset
    gold = {q.id: q for q in dataset.load(dataset_name)}
    schema = Schema()
    for path in paths:
        rows = [r for r in json.load(open(path)) if r.get("trace")]
        if not rows:
            print(f"{os.path.basename(path)}: no pipeline traces, skipped")
            continue
        cfg = rows[0]["trace"]["config"]
        n = len(rows)
        ins = [r for r in rows if in_scope(r["question"])]
        out = [r for r in rows if not in_scope(r["question"])]
        acc = lambda rs: sum(r["correct"] for r in rs)  # noqa: E731

        r_ok = r_fp = r_fn = 0
        rec_num = rec_den = prec_num = prec_den = 0
        rep_needed = rep_fixed = 0
        for r in rows:
            q = gold[r["id"]]
            said = r["trace"]["router"]["answerable"]
            r_ok += said == q.answerable
            r_fn += q.answerable and not said
            r_fp += (not q.answerable) and said
            ln = r["trace"].get("linker")
            if q.answerable and ln is not None:
                want = gold_tables(q.gold_sql, schema.tables)
                got = set(ln["tables"])
                rec_num += len(want & got)
                rec_den += len(want)
                prec_num += len(want & got)
                prec_den += len(got)
            rp = r["trace"].get("repair") or {}
            if "repaired" in rp:
                rep_needed += 1
                rep_fixed += bool(rp["repaired"])

        print(f"\n{os.path.basename(path)}  {cfg}")
        print(f"  end to end   {pct(acc(rows), n)}% strict  {pct(sum(r['loose'] for r in rows), n)}% loose"
              f"   in-scope {acc(ins)}/{len(ins)}   out-of-scope {acc(out)}/{len(out)}")
        print(f"  router       {pct(r_ok, n)}% correct   refused answerable {r_fn}"
              f"   let unanswerable through {r_fp}")
        print(f"  linker       recall {pct(rec_num, rec_den)}%   precision {pct(prec_num, prec_den)}%")
        print(f"  repair       needed {rep_needed}   fixed {rep_fixed}")
        print(f"  cost/q       {sum(r['model_calls'] for r in rows) / n:.1f} calls"
              f"   {sum(r['tokens_in'] + r['tokens_out'] for r in rows) / n:.0f} tokens"
              f"   {sum(r['secs'] for r in rows) / n:.2f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    ds = "chinook_v2"
    if args and args[0].startswith("--dataset="):
        ds = args.pop(0).split("=", 1)[1]
    if not args:
        sys.exit(__doc__)
    main(args, ds)

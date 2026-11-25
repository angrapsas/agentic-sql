#!/usr/bin/env python3
"""Did a model on the renamed schema write SQL using the ORIGINAL Chinook names?

The renamed database has no table called Track and no column called InvoiceId.
A model that uses those names was not reading the schema it was shown: it was
writing from memory of Chinook. A model that finds the task harder for other
reasons (the question says "invoice", the table is now Sale) makes different
mistakes: wrong joins or wrong columns, but using the new names.

For each arm, counts questions whose final SQL, or any query it ran along the
way, uses an original-only name. Name and Title are skipped: they exist in both.

    python scripts/recall_check.py results/renamed/rn-haiku-oneshot.json [more.json ...]
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAP = json.load(open(os.path.join(HERE, "..", "data", "chinook_v2_renamed.json")))["rename_map"]
OLD = sorted(set(MAP["tables"]) | set(MAP["columns"]), key=len, reverse=True)
PATTERN = re.compile(r"\b(" + "|".join(OLD) + r")\b")


def old_names(text):
    # drop string literals first, so a value like 'Track 5' does not count
    text = re.sub(r"'(?:[^']|'')*'", "''", text or "")
    return sorted(set(PATTERN.findall(text)))


def main(paths):
    for path in paths:
        rows = json.load(open(path))
        if isinstance(rows, dict):
            rows = rows.get("results", rows.get("rows", []))
        wrong = [r for r in rows if not r.get("correct")]
        hits, names = 0, {}
        for r in rows:
            found = set(old_names(r.get("sql", "")))
            for q in r.get("queries_list", []) or []:
                found |= set(old_names(q if isinstance(q, str) else json.dumps(q)))
            if found:
                hits += 1
                for n in found:
                    names[n] = names.get(n, 0) + 1
        top = ", ".join(f"{n} {c}" for n, c in sorted(names.items(), key=lambda x: -x[1])[:8])
        print(f"{os.path.basename(path)}: {len(rows)} questions, {len(wrong)} wrong, "
              f"{hits} used an original Chinook name")
        if top:
            print(f"    most used: {top}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])

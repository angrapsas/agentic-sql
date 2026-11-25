#!/usr/bin/env python3
"""Renamed Chinook: same data, same questions, different table and column names.

The question it answers: does a model read the schema it is given, or recall
Chinook from pre-training? Names are replaced with different but meaningful
ones (Track -> Song, Composer -> Writer), not gibberish, so the task stays as
hard as before for a model that reads the schema. Only recall is removed.
Name and Title are left alone: they are generic, and renaming them adds nothing.

Two steps:

  python data/build_renamed.py db
      Builds data/Chinook_renamed.db from data/Chinook.db. Stdlib only. Run this
      on every machine, after data/verify_dataset.py.

  python data/build_renamed.py dataset
      Rewrites the gold SQL of chinook_v2 and train_v1 with the same map
      (needs sqlglot) and checks every rewritten query returns exactly the rows
      the original returns on the original database. Writes
      chinook_v2_renamed.json and train_v1_renamed.jsonl. Rerun after any
      change to chinook_v2.

Use the renamed files together, by pointing the harness at the renamed db:
  AGSQL_DB=data/Chinook_renamed.db python -m agsql.cli run ...
The dataset records which db it needs, and the CLI refuses a mismatch.
"""

import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DB = os.path.join(HERE, "Chinook.db")
NEW_DB = os.path.join(HERE, "Chinook_renamed.db")

TABLES = {
    "Artist": "Performer", "Album": "Record", "Track": "Song", "Genre": "Style",
    "MediaType": "Format", "Playlist": "Mix", "PlaylistTrack": "MixSong",
    "Customer": "Client", "Employee": "Staff", "Invoice": "Sale",
    "InvoiceLine": "SaleItem",
}
COLUMNS = {
    "ArtistId": "PerformerId", "AlbumId": "RecordId", "TrackId": "SongId",
    "GenreId": "StyleId", "MediaTypeId": "FormatId", "PlaylistId": "MixId",
    "CustomerId": "ClientId", "EmployeeId": "StaffId", "InvoiceId": "SaleId",
    "InvoiceLineId": "SaleItemId", "SupportRepId": "AccountManagerId",
    "ReportsTo": "ManagerId", "Composer": "Writer", "Milliseconds": "DurationMs",
    "Bytes": "SizeBytes", "UnitPrice": "Price", "Quantity": "Qty",
    "Total": "Amount", "InvoiceDate": "SaleDate", "BillingAddress": "BillAddress",
    "BillingCity": "BillCity", "BillingState": "BillState",
    "BillingCountry": "BillCountry", "BillingPostalCode": "BillPostcode",
    "FirstName": "GivenName", "LastName": "FamilyName", "Company": "Employer",
    "HireDate": "StartDate", "BirthDate": "DateOfBirth", "Email": "EmailAddress",
    "Phone": "PhoneNumber", "Fax": "FaxNumber", "PostalCode": "Postcode",
}


def build_db():
    if not os.path.exists(SRC_DB):
        sys.exit("data/Chinook.db missing: run python data/verify_dataset.py first")
    # Build beside the real file and swap it in with one atomic rename: the folder
    # is shared between clusters, and a job may be reading the database right now.
    tmp = NEW_DB + f".tmp{os.getpid()}"
    if os.path.exists(tmp):
        os.remove(tmp)
    src = sqlite3.connect(SRC_DB)
    dst = sqlite3.connect(tmp)
    src.backup(dst)
    src.close()
    dst.execute("PRAGMA legacy_alter_table = OFF")
    for t in [r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
        for col in [r[1] for r in dst.execute(f'PRAGMA table_info("{t}")')]:
            if col in COLUMNS:
                dst.execute(f'ALTER TABLE "{t}" RENAME COLUMN "{col}" TO "{COLUMNS[col]}"')
    for old, new in TABLES.items():
        dst.execute(f'ALTER TABLE "{old}" RENAME TO "{new}"')
    dst.commit()
    # nothing of the old schema may survive
    leftovers = []
    for t in [r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
        if t in TABLES:
            leftovers.append(t)
        leftovers += [f"{t}.{c[1]}" for c in dst.execute(f'PRAGMA table_info("{t}")') if c[1] in COLUMNS]
    dst.close()
    if leftovers:
        os.remove(tmp)
        sys.exit(f"old names survived: {leftovers}")
    os.replace(tmp, NEW_DB)
    print(f"wrote {NEW_DB}")


def rewrite_sql(sql):
    import sqlglot
    from sqlglot import exp
    tree = sqlglot.parse_one(sql, read="sqlite")
    for t in tree.find_all(exp.Table):
        if t.name in TABLES:
            t.set("this", exp.to_identifier(TABLES[t.name]))
    for c in tree.find_all(exp.Column):
        if c.name in COLUMNS:
            c.set("this", exp.to_identifier(COLUMNS[c.name]))
    return tree.sql(dialect="sqlite")


def rewrite_text(text):
    # identifiers that appear verbatim in question text, e.g. "AlbumId 100"
    for old, new in {**COLUMNS, **TABLES}.items():
        if re.search(r"[a-z][A-Z]", old):  # only CamelCase identifiers, not plain words
            text = re.sub(rf"\b{old}\b", new, text)
    return text


def rows(db, sql):
    con = sqlite3.connect(db)
    try:
        return sorted(map(repr, con.execute(sql).fetchall()))
    finally:
        con.close()


def build_dataset():
    if not os.path.exists(NEW_DB):
        build_db()
    problems = []

    v2 = json.load(open(os.path.join(HERE, "chinook_v2.json")))
    out = []
    for q in v2["questions"]:
        q = dict(q)
        q["question"] = rewrite_text(q["question"])
        if q["answerable"]:
            new = rewrite_sql(q["gold_sql"])
            if rows(SRC_DB, q["gold_sql"]) != rows(NEW_DB, new):
                problems.append(f"{q['id']}: rewritten gold returns different rows")
            q["gold_sql"] = new
        out.append(q)
    meta = dict(v2, name="chinook_v2_renamed", db_file="Chinook_renamed.db",
                questions=out, rename_map={"tables": TABLES, "columns": COLUMNS})
    with open(os.path.join(HERE, "chinook_v2_renamed.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    train_out = []
    for line in open(os.path.join(HERE, "train_v1.jsonl")):
        p = json.loads(line)
        new = rewrite_sql(p["sql"])
        if rows(SRC_DB, p["sql"]) != rows(NEW_DB, new):
            problems.append(f"train: rewritten SQL differs for: {p['question']}")
        train_out.append(dict(p, question=rewrite_text(p["question"]), sql=new))
    with open(os.path.join(HERE, "train_v1_renamed.jsonl"), "w") as f:
        for p in train_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s)")
    print(f"{len(out)} questions and {len(train_out)} training pairs rewritten; "
          "every rewritten query returns exactly the original rows")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"db": build_db, "dataset": build_dataset}.get(step, lambda: sys.exit(__doc__))()

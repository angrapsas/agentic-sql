#!/usr/bin/env python3
"""Check chinook_v1 against a freshly built database.

Downloads the Chinook SQL, builds the database, then asserts that every gold
query still parses, executes, returns a non-empty result, and returns the row
count the dataset records. Run this before trusting any result produced with
this dataset, and after any change to it.

    python verify_dataset.py
"""

import json
import os
import sqlite3
import sys
import urllib.request

SQL_URL = ("https://raw.githubusercontent.com/lerocha/chinook-database/master/"
           "ChinookDatabase/DataSources/Chinook_Sqlite.sql")
HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "Chinook.db")
DATA = os.path.join(HERE, "chinook_v1.json")

EXPECTED_TABLES = {"Album", "Artist", "Customer", "Employee", "Genre", "Invoice",
                   "InvoiceLine", "MediaType", "Playlist", "PlaylistTrack", "Track"}
EXPECTED_COUNTS = {"Track": 3503, "Artist": 275, "Customer": 59, "Invoice": 412}


def build_db():
    if os.path.exists(DB):
        print(f"using existing {DB}")
        return
    sql_path = os.path.join(HERE, "chinook.sql")
    if not os.path.exists(sql_path):
        print("downloading Chinook SQL")
        urllib.request.urlretrieve(SQL_URL, sql_path)
    print("building database")
    con = sqlite3.connect(DB)
    # utf-8-sig: the upstream file carries a BOM
    con.executescript(open(sql_path, encoding="utf-8-sig").read())
    con.commit()
    con.close()


def main():
    build_db()
    con = sqlite3.connect(DB)
    data = json.load(open(DATA))
    failures = []

    tables = {r[0] for r in con.execute(
        "select name from sqlite_master where type='table'")}
    if tables != EXPECTED_TABLES:
        failures.append(f"table set differs: missing {EXPECTED_TABLES - tables}, "
                        f"extra {tables - EXPECTED_TABLES}")
    for t, n in EXPECTED_COUNTS.items():
        got = con.execute(f'select count(*) from "{t}"').fetchone()[0]
        if got != n:
            failures.append(f"{t} has {got} rows, expected {n}")

    answerable = unanswerable = 0
    for q in data["questions"]:
        qid, sql = q["id"], q["gold_sql"]
        if not q["answerable"]:
            unanswerable += 1
            if sql:
                failures.append(f"{qid}: marked unanswerable but has a gold query")
            continue
        answerable += 1
        if not sql:
            failures.append(f"{qid}: answerable but has no gold query")
            continue
        try:
            rows = con.execute(sql).fetchall()
        except Exception as e:
            failures.append(f"{qid}: gold query failed: {e}")
            continue
        if not rows:
            failures.append(f"{qid}: gold query returns no rows")
        elif q["gold_row_count"] not in ("", None) and len(rows) != int(q["gold_row_count"]):
            failures.append(f"{qid}: returns {len(rows)} rows, dataset records "
                            f"{q['gold_row_count']}")

    ids = [q["id"] for q in data["questions"]]
    if len(ids) != len(set(ids)):
        failures.append("duplicate ids")
    if len(ids) != data["n"]:
        failures.append(f"n says {data['n']} but file holds {len(ids)}")
    questions = [q["question"] for q in data["questions"]]
    if len(questions) != len(set(questions)):
        failures.append("duplicate questions")

    con.close()

    print(f"\n{len(ids)} questions: {answerable} answerable, {unanswerable} unanswerable")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("all gold queries execute and match their recorded row counts")
    print("OK")


if __name__ == "__main__":
    main()

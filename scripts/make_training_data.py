#!/usr/bin/env python3
"""Generate training pairs for the adapter, and verify every one of them.

Each pair is a question and a SQL query that answers it. Questions are built by
filling templates with real values pulled from the database, so the vocabulary
the adapter sees is the vocabulary the schema actually uses.

Two rules the generator enforces, both checked rather than assumed:

  Every query must execute and return at least one row. A pair whose query
  errors or returns nothing teaches the model nothing useful.

  No pair may duplicate a question in the evaluation set. Overlap would make the
  evaluation meaningless, so the check is on the question text and on the
  normalised SQL.

    python scripts/make_training_data.py --out data/train_v1.jsonl
"""

import argparse
import json
import os
import random
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

DB = os.path.join(ROOT, "data", "Chinook.db")
EVAL = os.path.join(ROOT, "data", "chinook_v1.json")


def norm_sql(s):
    return re.sub(r"\s+", " ", s or "").strip().lower().rstrip(";")


def sample(con, sql, n, seed=0):
    rows = [r[0] for r in con.execute(sql) if r[0] not in (None, "")]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def build(con):
    """Returns a list of (question, sql) pairs. Templates are grouped by the
    kind of query they teach, not by topic."""
    pairs = []
    add = lambda q, s: pairs.append((q, s))

    artists = sample(con, "SELECT Name FROM Artist", 14, 1)
    genres = sample(con, "SELECT Name FROM Genre", 12, 2)
    countries = sample(con, "SELECT DISTINCT Country FROM Customer", 12, 3)
    cities = sample(con, "SELECT DISTINCT City FROM Customer", 10, 4)
    albums = sample(con, "SELECT Title FROM Album", 12, 5)
    playlists = sample(con, "SELECT DISTINCT Name FROM Playlist", 8, 6)
    media = sample(con, "SELECT Name FROM MediaType", 5, 7)
    tracks = sample(con,
                    "SELECT Name FROM Track WHERE Composer IS NOT NULL "
                    "AND Name NOT LIKE '%''%' GROUP BY Name HAVING COUNT(*) = 1",
                    14, 8)

    q = lambda s: s.replace("'", "''")

    # single table lookups and filters
    for t in tracks:
        add(f"Who wrote the track '{t}'?",
            f"SELECT Composer FROM Track WHERE Name = '{q(t)}'")
        add(f"How long is the track '{t}' in milliseconds?",
            f"SELECT Milliseconds FROM Track WHERE Name = '{q(t)}'")
    for a in albums:
        add(f"Which artist released the album '{a}'?",
            "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
            f"WHERE al.Title = '{q(a)}'")
    for c in countries:
        add(f"How many customers live in {c}?",
            f"SELECT COUNT(*) FROM Customer WHERE Country = '{q(c)}'")
        add(f"What is the total amount invoiced to {c}?",
            f"SELECT SUM(Total) FROM Invoice WHERE BillingCountry = '{q(c)}'")
    for c in cities:
        add(f"Which customers are based in {c}?",
            f"SELECT FirstName, LastName FROM Customer WHERE City = '{q(c)}'")

    # joins across two or three tables
    for g in genres:
        add(f"How many tracks are in the {g} genre?",
            "SELECT COUNT(*) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId "
            f"WHERE g.Name = '{q(g)}'")
        add(f"What is the average length of a {g} track in milliseconds?",
            "SELECT AVG(t.Milliseconds) FROM Track t JOIN Genre g "
            f"ON g.GenreId = t.GenreId WHERE g.Name = '{q(g)}'")
    for a in artists:
        add(f"How many albums has {a} released?",
            "SELECT COUNT(*) FROM Album al JOIN Artist ar "
            f"ON ar.ArtistId = al.ArtistId WHERE ar.Name = '{q(a)}'")
        add(f"How many tracks does {a} have?",
            "SELECT COUNT(*) FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId "
            f"JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = '{q(a)}'")
    for p in playlists:
        add(f"How many tracks are on the {p} playlist?",
            "SELECT COUNT(*) FROM PlaylistTrack pt JOIN Playlist p "
            f"ON p.PlaylistId = pt.PlaylistId WHERE p.Name = '{q(p)}'")
    for m in media:
        add(f"How many tracks use the {m} format?",
            "SELECT COUNT(*) FROM Track t JOIN MediaType m "
            f"ON m.MediaTypeId = t.MediaTypeId WHERE m.Name = '{q(m)}'")

    # aggregates, superlatives and vocabulary the user might use instead
    add("How many songs are there in total?", "SELECT COUNT(*) FROM Track")
    add("How many different genres are there?", "SELECT COUNT(*) FROM Genre")
    add("How many employees work here?", "SELECT COUNT(*) FROM Employee")
    add("How many playlists exist?", "SELECT COUNT(*) FROM Playlist")
    add("What is the longest track in milliseconds?",
        "SELECT MAX(Milliseconds) FROM Track")
    add("What is the shortest track in milliseconds?",
        "SELECT MIN(Milliseconds) FROM Track")
    add("What is the most expensive track price?", "SELECT MAX(UnitPrice) FROM Track")
    add("What is the average amount spent per invoice?", "SELECT AVG(Total) FROM Invoice")
    add("What is the largest single invoice?", "SELECT MAX(Total) FROM Invoice")
    add("How many invoices are there altogether?", "SELECT COUNT(*) FROM Invoice")
    add("Which genre has the fewest tracks?",
        "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
        "GROUP BY g.GenreId ORDER BY COUNT(*) ASC LIMIT 1")
    add("Which album has the most tracks?",
        "SELECT al.Title FROM Album al JOIN Track t ON t.AlbumId = al.AlbumId "
        "GROUP BY al.AlbumId ORDER BY COUNT(*) DESC LIMIT 1")
    add("Which playlist has the most tracks?",
        "SELECT p.Name FROM Playlist p JOIN PlaylistTrack pt "
        "ON pt.PlaylistId = p.PlaylistId GROUP BY p.PlaylistId "
        "ORDER BY COUNT(*) DESC LIMIT 1")
    add("Which city has the most customers?",
        "SELECT City FROM Customer GROUP BY City ORDER BY COUNT(*) DESC LIMIT 1")
    add("Which employee supports the most customers?",
        "SELECT e.FirstName, e.LastName FROM Employee e JOIN Customer c "
        "ON c.SupportRepId = e.EmployeeId GROUP BY e.EmployeeId "
        "ORDER BY COUNT(*) DESC LIMIT 1")
    add("Which media type is used least often?",
        "SELECT m.Name FROM MediaType m JOIN Track t ON t.MediaTypeId = m.MediaTypeId "
        "GROUP BY m.MediaTypeId ORDER BY COUNT(*) ASC LIMIT 1")
    add("What is the total value of all sales?", "SELECT SUM(Total) FROM Invoice")
    add("How many tracks cost less than a dollar?",
        "SELECT COUNT(*) FROM Track WHERE UnitPrice < 1.0")
    add("How many customers do not have a company listed?",
        "SELECT COUNT(*) FROM Customer WHERE Company IS NULL")
    add("How many tracks are longer than ten minutes?",
        "SELECT COUNT(*) FROM Track WHERE Milliseconds > 600000")
    add("How many albums are there?", "SELECT COUNT(*) FROM Album")
    add("List the names of all the genres.", "SELECT Name FROM Genre")
    add("How many invoice line items are there?", "SELECT COUNT(*) FROM InvoiceLine")
    add("What countries do customers come from?",
        "SELECT DISTINCT Country FROM Customer")
    add("How many customers has each support rep been assigned?",
        "SELECT e.FirstName, COUNT(*) FROM Employee e JOIN Customer c "
        "ON c.SupportRepId = e.EmployeeId GROUP BY e.EmployeeId")
    add("Which tracks have no composer listed?",
        "SELECT Name FROM Track WHERE Composer IS NULL")
    add("How much revenue came from each country?",
        "SELECT BillingCountry, SUM(Total) FROM Invoice GROUP BY BillingCountry")
    add("How many tracks are on each media type?",
        "SELECT m.Name, COUNT(*) FROM MediaType m JOIN Track t "
        "ON t.MediaTypeId = m.MediaTypeId GROUP BY m.MediaTypeId")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "train_v1.jsonl"))
    ap.add_argument("--db", default=DB)
    args = ap.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"{args.db} not found. Run data/verify_dataset.py first.")

    con = sqlite3.connect(args.db)
    eval_data = json.load(open(EVAL))
    eval_questions = {q["question"].strip().lower() for q in eval_data["questions"]}
    eval_sql = {norm_sql(q["gold_sql"]) for q in eval_data["questions"] if q["gold_sql"]}

    pairs = build(con)
    kept, dropped = [], {"error": 0, "empty": 0, "overlap_q": 0, "overlap_sql": 0,
                         "duplicate": 0}
    seen = set()

    for question, sql in pairs:
        if question.strip().lower() in eval_questions:
            dropped["overlap_q"] += 1
            continue
        if norm_sql(sql) in eval_sql:
            dropped["overlap_sql"] += 1
            continue
        if question.strip().lower() in seen:
            dropped["duplicate"] += 1
            continue
        try:
            rows = con.execute(sql).fetchall()
        except Exception:
            dropped["error"] += 1
            continue
        if not rows:
            dropped["empty"] += 1
            continue
        seen.add(question.strip().lower())
        kept.append({"question": question, "sql": sql})

    con.close()

    with open(args.out, "w") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")

    print(f"generated {len(pairs)} candidates, kept {len(kept)}")
    for k, v in dropped.items():
        if v:
            print(f"  dropped {v} ({k})")
    print(f"wrote {args.out}")

    if len(kept) < 80:
        print("\nWARNING: fewer than 80 pairs. The adapter has little to learn from.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

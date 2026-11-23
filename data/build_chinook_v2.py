#!/usr/bin/env python3
"""Build chinook_v2: the 43 chinook_v1 questions, unchanged, plus new ones.

DRAFT. Every new question below needs a human read before v2 is frozen: is it
unambiguous (or ambiguous on purpose, with the gold covering every reading), is
the gold what a careful analyst would return, and is it in the right bucket.

The script refuses to write the dataset if any check fails:
  - every gold query executes and returns at least one row
  - superlatives marked unique really have a unique winner
  - no question or normalised gold SQL overlaps the training pairs
  - no question repeats a v1 question
v1 ids, questions and gold are copied verbatim, so v1 results stay comparable
on the v1 subset.

    python data/build_chinook_v2.py           # check and write chinook_v2.json/.csv
    python data/build_chinook_v2.py --check   # check only
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "Chinook.db")
V1 = os.path.join(HERE, "chinook_v1.json")
TRAIN = os.path.join(HERE, "train_v1.jsonl")
OUT = os.path.join(HERE, "chinook_v2")

# (category_id, question, gold_sql or None, notes, unique_check)
# unique_check: for a superlative answered with LIMIT 1, a query returning the
# ranking values in order; the build fails if the top two are equal.
NEW = [
    # ------------------------------------------------------------- vocabulary
    # 1 single-table lookup
    (1, "What is the email address of the customer Luís Gonçalves?",
     "SELECT Email FROM Customer WHERE FirstName = 'Luís' AND LastName = 'Gonçalves'", "", None),
    (1, "What company does the customer Frank Harris work for?",
     "SELECT Company FROM Customer WHERE FirstName = 'Frank' AND LastName = 'Harris'", "", None),
    (1, "On what date was Steve Johnson hired?",
     "SELECT HireDate FROM Employee WHERE FirstName = 'Steve' AND LastName = 'Johnson'", "", None),
    (1, "What is Laura Callahan's job title?",
     "SELECT Title FROM Employee WHERE FirstName = 'Laura' AND LastName = 'Callahan'", "", None),
    (1, "Which city was invoice 100 billed to?",
     "SELECT BillingCity FROM Invoice WHERE InvoiceId = 100", "", None),
    (1, "What is the phone number on file for the customer Helena Holý?",
     "SELECT Phone FROM Customer WHERE FirstName = 'Helena' AND LastName = 'Holý'", "", None),
    # 2 filtered count
    (2, "How many employees are based in Calgary?",
     "SELECT COUNT(*) FROM Employee WHERE City = 'Calgary'", "", None),
    (2, "How many tracks run over six minutes?",
     "SELECT COUNT(*) FROM Track WHERE Milliseconds > 360000", "", None),
    (2, "How many invoices came to more than 15 dollars?",
     "SELECT COUNT(*) FROM Invoice WHERE Total > 15", "", None),
    (2, "How many customers have a fax number on file?",
     "SELECT COUNT(*) FROM Customer WHERE Fax IS NOT NULL", "", None),
    (2, "How many customers have an address in the state of California?",
     "SELECT COUNT(*) FROM Customer WHERE State = 'CA'",
     "State is stored as the postal code, 'CA'.", None),
    (2, "How many tracks are priced at 1.99?",
     "SELECT COUNT(*) FROM Track WHERE UnitPrice = 1.99", "", None),
    (2, "How many invoices were sent to Canadian billing addresses?",
     "SELECT COUNT(*) FROM Invoice WHERE BillingCountry = 'Canada'", "", None),
    # 3 aggregate
    (3, "What is the combined size in bytes of every track?",
     "SELECT SUM(Bytes) FROM Track", "", None),
    (3, "What is the average unit price of a track?",
     "SELECT AVG(UnitPrice) FROM Track", "", None),
    (3, "What is the smallest invoice total on record?",
     "SELECT MIN(Total) FROM Invoice", "", None),
    (3, "How many distinct composers are credited across all tracks?",
     "SELECT COUNT(DISTINCT Composer) FROM Track", "", None),
    (3, "How many different countries appear as billing countries on invoices?",
     "SELECT COUNT(DISTINCT BillingCountry) FROM Invoice", "", None),
    (3, "What is the average number of bytes per track?",
     "SELECT AVG(Bytes) FROM Track", "", None),
    # 5 multi-hop join (3+ tables)
    (5, "Which genres does AC/DC have tracks in?",
     "SELECT DISTINCT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
     "JOIN Album al ON al.AlbumId = t.AlbumId JOIN Artist ar ON ar.ArtistId = al.ArtistId "
     "WHERE ar.Name = 'AC/DC'", "", None),
    (5, "How many minutes of music does Pearl Jam have in the catalogue?",
     "SELECT SUM(t.Milliseconds) / 60000.0 FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId "
     "JOIN Artist ar ON ar.ArtistId = al.ArtistId WHERE ar.Name = 'Pearl Jam'",
     "Real division. Integer division truncates.", None),
    (5, "How many invoice lines were for tracks in the Jazz genre?",
     "SELECT COUNT(*) FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId "
     "JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Jazz'", "", None),
    (5, "How much did customers in France spend on Metal tracks?",
     "SELECT SUM(il.UnitPrice * il.Quantity) FROM InvoiceLine il JOIN Invoice i ON i.InvoiceId = il.InvoiceId "
     "JOIN Customer c ON c.CustomerId = i.CustomerId JOIN Track t ON t.TrackId = il.TrackId "
     "JOIN Genre g ON g.GenreId = t.GenreId WHERE c.Country = 'France' AND g.Name = 'Metal'", "", None),
    (5, "Which playlists include the track 'Enter Sandman'?",
     "SELECT DISTINCT p.Name FROM Playlist p JOIN PlaylistTrack pt ON pt.PlaylistId = p.PlaylistId "
     "JOIN Track t ON t.TrackId = pt.TrackId WHERE t.Name = 'Enter Sandman'",
     "Distinct names: several playlists share a name.", None),
    (5, "Which customers have bought a track from the album 'Let There Be Rock'?",
     "SELECT DISTINCT c.FirstName, c.LastName FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
     "JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId JOIN Track t ON t.TrackId = il.TrackId "
     "JOIN Album al ON al.AlbumId = t.AlbumId WHERE al.Title = 'Let There Be Rock'", "", None),
    (5, "Which support rep looks after the customer who lives in Prague and is named Helena Holý?",
     "SELECT e.FirstName, e.LastName FROM Employee e JOIN Customer c ON c.SupportRepId = e.EmployeeId "
     "WHERE c.FirstName = 'Helena' AND c.LastName = 'Holý'",
     "Two tables only; kept here as a join through a foreign key with a non-obvious name.", None),
    (5, "How many tracks by Metallica have ever been sold?",
     "SELECT COUNT(DISTINCT t.TrackId) FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId "
     "JOIN Artist ar ON ar.ArtistId = al.ArtistId JOIN InvoiceLine il ON il.TrackId = t.TrackId "
     "WHERE ar.Name = 'Metallica'", "Distinct tracks, not sales.", None),
    # 14 vocabulary mismatch
    (14, "How many bands have more than ten records out?",
     "SELECT COUNT(*) FROM (SELECT ArtistId FROM Album GROUP BY ArtistId HAVING COUNT(*) > 10)",
     "bands = Artist, records = Album.", None),
    (14, "What is the total playing time of the whole catalogue, in hours?",
     "SELECT SUM(Milliseconds) / 3600000.0 FROM Track", "Real division.", None),
    (14, "How many shoppers are registered with the store?",
     "SELECT COUNT(*) FROM Customer", "shoppers = Customer.", None),
    (14, "How many tunes last longer than seven minutes?",
     "SELECT COUNT(*) FROM Track WHERE Milliseconds > 420000", "tunes = Track.", None),
    (14, "Which staff members look after customer accounts?",
     "SELECT DISTINCT e.FirstName, e.LastName FROM Employee e JOIN Customer c ON c.SupportRepId = e.EmployeeId",
     "look after accounts = SupportRepId.", None),
    (14, "How many workers joined the company in 2003?",
     "SELECT COUNT(*) FROM Employee WHERE HireDate >= '2003-01-01' AND HireDate < '2004-01-01'",
     "workers = Employee, joined = HireDate.", None),
    (14, "How many orders included more than ten items?",
     "SELECT COUNT(*) FROM (SELECT InvoiceId FROM InvoiceLine GROUP BY InvoiceId HAVING SUM(Quantity) > 10)",
     "orders = Invoice, items = InvoiceLine quantity.", None),

    # ------------------------------------------------------------------ logic
    # 4 superlative
    (4, "Which artist has the most albums?",
     "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
     "GROUP BY ar.ArtistId ORDER BY COUNT(*) DESC LIMIT 1", "",
     "SELECT COUNT(*) FROM Album GROUP BY ArtistId ORDER BY 1 DESC LIMIT 2"),
    (4, "Who is the oldest employee?",
     "SELECT FirstName, LastName FROM Employee ORDER BY BirthDate LIMIT 1", "",
     "SELECT BirthDate FROM Employee ORDER BY 1 LIMIT 2"),
    (4, "Which customer placed the single largest invoice?",
     "SELECT c.FirstName, c.LastName FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
     "ORDER BY i.Total DESC LIMIT 1", "",
     "SELECT Total FROM Invoice ORDER BY 1 DESC LIMIT 2"),
    (4, "Which album has the longest total running time?",
     "SELECT al.Title FROM Album al JOIN Track t ON t.AlbumId = al.AlbumId "
     "GROUP BY al.AlbumId ORDER BY SUM(t.Milliseconds) DESC LIMIT 1", "",
     "SELECT SUM(Milliseconds) FROM Track GROUP BY AlbumId ORDER BY 1 DESC LIMIT 2"),
    (4, "Which media type has brought in the most revenue?",
     "SELECT m.Name FROM MediaType m JOIN Track t ON t.MediaTypeId = m.MediaTypeId "
     "JOIN InvoiceLine il ON il.TrackId = t.TrackId GROUP BY m.MediaTypeId "
     "ORDER BY SUM(il.UnitPrice * il.Quantity) DESC LIMIT 1", "",
     "SELECT SUM(il.UnitPrice * il.Quantity) FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId "
     "GROUP BY t.MediaTypeId ORDER BY 1 DESC LIMIT 2"),
    (4, "Whose customers have spent the most in total, among the support agents?",
     "SELECT e.FirstName, e.LastName FROM Employee e JOIN Customer c ON c.SupportRepId = e.EmployeeId "
     "JOIN Invoice i ON i.CustomerId = c.CustomerId GROUP BY e.EmployeeId ORDER BY SUM(i.Total) DESC LIMIT 1", "",
     "SELECT SUM(i.Total) FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
     "GROUP BY c.SupportRepId ORDER BY 1 DESC LIMIT 2"),
    (4, "Which genre has the longest average track length?",
     "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
     "GROUP BY g.GenreId ORDER BY AVG(t.Milliseconds) DESC LIMIT 1", "",
     "SELECT AVG(Milliseconds) FROM Track GROUP BY GenreId ORDER BY 1 DESC LIMIT 2"),
    (4, "Which track is the shortest?",
     "SELECT Name FROM Track ORDER BY Milliseconds LIMIT 1", "",
     "SELECT Milliseconds FROM Track ORDER BY 1 LIMIT 2"),
    # 6 self-join
    (6, "Who manages the IT staff?",
     "SELECT DISTINCT m.FirstName, m.LastName FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "WHERE e.Title = 'IT Staff'", "", None),
    (6, "How many people report directly to Andrew Adams?",
     "SELECT COUNT(*) FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "WHERE m.FirstName = 'Andrew' AND m.LastName = 'Adams'", "", None),
    (6, "Which employees share a manager with Jane Peacock?",
     "SELECT e.FirstName, e.LastName FROM Employee e JOIN Employee j ON e.ReportsTo = j.ReportsTo "
     "WHERE j.FirstName = 'Jane' AND j.LastName = 'Peacock' AND e.EmployeeId <> j.EmployeeId",
     "Excludes Jane herself.", None),
    (6, "Which employees were hired before their own manager?",
     "SELECT e.FirstName, e.LastName FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "WHERE e.HireDate < m.HireDate", "", None),
    (6, "Which employees are older than the person they report to?",
     "SELECT e.FirstName, e.LastName FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "WHERE e.BirthDate < m.BirthDate", "Older = earlier BirthDate.", None),
    (6, "Who is Robert King's manager's manager?",
     "SELECT g.FirstName, g.LastName FROM Employee e JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "JOIN Employee g ON m.ReportsTo = g.EmployeeId WHERE e.FirstName = 'Robert' AND e.LastName = 'King'",
     "", None),
    # 7 anti-join
    (7, "How many genres have never had a single sale?",
     "SELECT COUNT(*) FROM Genre g WHERE NOT EXISTS (SELECT 1 FROM Track t JOIN InvoiceLine il "
     "ON il.TrackId = t.TrackId WHERE t.GenreId = g.GenreId)", "", None),
    (7, "How many playlists contain no tracks at all?",
     "SELECT COUNT(*) FROM Playlist p WHERE NOT EXISTS (SELECT 1 FROM PlaylistTrack pt WHERE pt.PlaylistId = p.PlaylistId)",
     "", None),
    (7, "How many customers have never bought a Rock track?",
     "SELECT COUNT(*) FROM Customer c WHERE NOT EXISTS (SELECT 1 FROM Invoice i JOIN InvoiceLine il "
     "ON il.InvoiceId = i.InvoiceId JOIN Track t ON t.TrackId = il.TrackId JOIN Genre g ON g.GenreId = t.GenreId "
     "WHERE i.CustomerId = c.CustomerId AND g.Name = 'Rock')", "", None),
    (7, "How many albums have never had any of their tracks sold?",
     "SELECT COUNT(*) FROM Album al WHERE NOT EXISTS (SELECT 1 FROM Track t JOIN InvoiceLine il "
     "ON il.TrackId = t.TrackId WHERE t.AlbumId = al.AlbumId)", "", None),
    (7, "Which employees are not the support rep for any customer?",
     "SELECT FirstName, LastName FROM Employee e WHERE NOT EXISTS (SELECT 1 FROM Customer c WHERE c.SupportRepId = e.EmployeeId)",
     "", None),
    (7, "How many artists have never had a track sold?",
     "SELECT COUNT(*) FROM Artist ar WHERE NOT EXISTS (SELECT 1 FROM Album al JOIN Track t ON t.AlbumId = al.AlbumId "
     "JOIN InvoiceLine il ON il.TrackId = t.TrackId WHERE al.ArtistId = ar.ArtistId)",
     "Includes artists with no albums.", None),
    # 9 temporal
    (9, "How many invoices were issued in the first quarter of 2022?",
     "SELECT COUNT(*) FROM Invoice WHERE InvoiceDate >= '2022-01-01' AND InvoiceDate < '2022-04-01'", "", None),
    (9, "What was total revenue in December 2024?",
     "SELECT SUM(Total) FROM Invoice WHERE InvoiceDate >= '2024-12-01' AND InvoiceDate < '2025-01-01'", "", None),
    (9, "Which year had the fewest invoices?",
     "SELECT STRFTIME('%Y', InvoiceDate) FROM Invoice GROUP BY 1 ORDER BY COUNT(*) LIMIT 1", "",
     "SELECT COUNT(*) FROM Invoice GROUP BY STRFTIME('%Y', InvoiceDate) ORDER BY 1 LIMIT 2"),
    (9, "How many different customers bought something in 2021?",
     "SELECT COUNT(DISTINCT CustomerId) FROM Invoice WHERE InvoiceDate >= '2021-01-01' AND InvoiceDate < '2022-01-01'",
     "", None),
    (9, "How many employees were born in the 1970s?",
     "SELECT COUNT(*) FROM Employee WHERE BirthDate >= '1970-01-01' AND BirthDate < '1980-01-01'", "", None),
    (9, "How much revenue came in during the second half of 2023?",
     "SELECT SUM(Total) FROM Invoice WHERE InvoiceDate >= '2023-07-01' AND InvoiceDate < '2024-01-01'", "", None),
    # 10 nested comparison
    (10, "How many invoices are larger than the average invoice?",
     "SELECT COUNT(*) FROM Invoice WHERE Total > (SELECT AVG(Total) FROM Invoice)", "", None),
    (10, "How many albums have more tracks than the average album?",
     "SELECT COUNT(*) FROM (SELECT AlbumId, COUNT(*) n FROM Track GROUP BY AlbumId) "
     "WHERE n > (SELECT AVG(n2) FROM (SELECT COUNT(*) n2 FROM Track GROUP BY AlbumId))", "", None),
    (10, "Which genres have an average track length above the overall average?",
     "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId GROUP BY g.GenreId "
     "HAVING AVG(t.Milliseconds) > (SELECT AVG(Milliseconds) FROM Track)", "", None),
    (10, "How many tracks cost more than the average track price?",
     "SELECT COUNT(*) FROM Track WHERE UnitPrice > (SELECT AVG(UnitPrice) FROM Track)", "", None),
    (10, "How many countries have more customers than the average country does?",
     "SELECT COUNT(*) FROM (SELECT Country, COUNT(*) n FROM Customer GROUP BY Country) "
     "WHERE n > (SELECT AVG(n2) FROM (SELECT COUNT(*) n2 FROM Customer GROUP BY Country))", "", None),
    # 11 ratio
    (11, "What percentage of tracks are video files?",
     "SELECT 100.0 * SUM(CASE WHEN m.Name LIKE '%video%' THEN 1 ELSE 0 END) / COUNT(*) "
     "FROM Track t JOIN MediaType m ON m.MediaTypeId = t.MediaTypeId", "", None),
    (11, "What share of total revenue came from customers in the USA, as a percentage?",
     "SELECT 100.0 * SUM(CASE WHEN BillingCountry = 'USA' THEN Total ELSE 0 END) / SUM(Total) FROM Invoice",
     "Uses billing country, which matches customer country throughout.", None),
    (11, "On average, how many tracks does an album have?",
     "SELECT 1.0 * (SELECT COUNT(*) FROM Track) / (SELECT COUNT(*) FROM Album)", "Real division.", None),
    (11, "What percentage of invoices were billed outside the USA and Canada?",
     "SELECT 100.0 * SUM(CASE WHEN BillingCountry NOT IN ('USA', 'Canada') THEN 1 ELSE 0 END) / COUNT(*) FROM Invoice",
     "", None),
    (11, "What fraction of artists have at least one album?",
     "SELECT 1.0 * (SELECT COUNT(DISTINCT ArtistId) FROM Album) / (SELECT COUNT(*) FROM Artist)", "", None),
    (11, "What percentage of tracks in the catalogue have ever been sold?",
     "SELECT 100.0 * (SELECT COUNT(DISTINCT TrackId) FROM InvoiceLine) / (SELECT COUNT(*) FROM Track)", "", None),

    # --------------------------------------------------------------- judgment
    # 8 ambiguous entity: gold covers every reading, like v1
    (8, "Who composed 'Angel'?",
     "SELECT Composer FROM Track WHERE Name = 'Angel'",
     "Two tracks are named 'Angel' with different composers. Gold returns both.", None),
    (8, "How big is the file for 'Stairway To Heaven', in bytes?",
     "SELECT Bytes FROM Track WHERE Name = 'Stairway To Heaven'",
     "Three tracks share this name. Gold returns all three sizes.", None),
    (8, "Which album is 'All My Love' on?",
     "SELECT al.Title FROM Album al JOIN Track t ON t.AlbumId = al.AlbumId WHERE t.Name = 'All My Love'",
     "Two tracks named 'All My Love'. Gold returns both albums.", None),
    (8, "Which city does the customer named Frank live in?",
     "SELECT City FROM Customer WHERE FirstName = 'Frank'",
     "Two customers are named Frank. Gold returns both cities.", None),
    (8, "What is Mark's email address?",
     "SELECT Email FROM Customer WHERE FirstName = 'Mark'",
     "Two customers are named Mark. Gold returns both.", None),
    (8, "What is the duration in milliseconds of 'Angel Of Harlem'?",
     "SELECT Milliseconds FROM Track WHERE Name = 'Angel Of Harlem'",
     "Two tracks named 'Angel Of Harlem'. Gold returns both.", None),
    (12, "Which customer or customers in Paris have spent the most?",
     "SELECT c.FirstName, c.LastName FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
     "WHERE c.City = 'Paris' GROUP BY c.CustomerId HAVING SUM(i.Total) = (SELECT MAX(s) FROM "
     "(SELECT SUM(i2.Total) s FROM Customer c2 JOIN Invoice i2 ON i2.CustomerId = c2.CustomerId "
     "WHERE c2.City = 'Paris' GROUP BY c2.CustomerId))",
     "Both Paris customers spent exactly 38.62. Gold returns both.", None),
    (12, "Which year or years had the most invoices?",
     "SELECT STRFTIME('%Y', InvoiceDate) FROM Invoice GROUP BY 1 HAVING COUNT(*) = "
     "(SELECT MAX(n) FROM (SELECT COUNT(*) n FROM Invoice GROUP BY STRFTIME('%Y', InvoiceDate)))",
     "2021 to 2024 all have 83 invoices.", None),
    (12, "Which employees were hired on the same day as another employee?",
     "SELECT e.FirstName, e.LastName FROM Employee e WHERE EXISTS (SELECT 1 FROM Employee o "
     "WHERE o.HireDate = e.HireDate AND o.EmployeeId <> e.EmployeeId)", "", None),
    (12, "Which cities have the most customers?",
     "SELECT City FROM Customer GROUP BY City HAVING COUNT(*) = "
     "(SELECT MAX(n) FROM (SELECT COUNT(*) n FROM Customer GROUP BY City))",
     "Several cities tie at the maximum.", None),
    (12, "Which playlists have the most tracks?",
     "SELECT p.Name FROM Playlist p JOIN PlaylistTrack pt ON pt.PlaylistId = p.PlaylistId GROUP BY p.PlaylistId "
     "HAVING COUNT(*) = (SELECT MAX(n) FROM (SELECT COUNT(*) n FROM PlaylistTrack GROUP BY PlaylistId))",
     "Playlists 1 and 8 are both named 'Music'.", None),
    (12, "In 2023, which month or months had the most invoices?",
     "SELECT STRFTIME('%m', InvoiceDate) FROM Invoice WHERE InvoiceDate >= '2023-01-01' AND InvoiceDate < '2024-01-01' "
     "GROUP BY 1 HAVING COUNT(*) = (SELECT MAX(n) FROM (SELECT COUNT(*) n FROM Invoice "
     "WHERE InvoiceDate >= '2023-01-01' AND InvoiceDate < '2024-01-01' GROUP BY STRFTIME('%m', InvoiceDate)))", "", None),
    (13, "What is the date of birth of the customer Luís Gonçalves?", None,
     "Customer has no birth date column. Employee does; do not confuse them.", None),
    (13, "How many followers does U2 have?", None, "No social data.", None),
    (13, "Which record label released 'Let There Be Rock'?", None, "No label data.", None),
    (13, "What star rating do listeners give 'Enter Sandman'?", None, "No ratings.", None),
    (13, "What musical key is 'Stairway To Heaven' in?", None, "No musical metadata.", None),
    (13, "What is Nancy Edwards's salary?", None, "No salary data.", None),
    (13, "Which customer has written the most product reviews?", None, "No reviews.", None),
    (13, "How much has the store paid Iron Maiden in royalties?", None, "No royalty data.", None),
    (13, "Which country does the artist Djavan come from?", None,
     "Artist has only a name. Customer countries are not artist countries.", None),
    (13, "What discount code did Frank Harris use on his last order?", None, "No discount data.", None),
    (13, "How many tracks are marked as explicit?", None, "No explicit flag.", None),
    (13, "What did shipping cost on invoice 100?", None, "Invoice has only a Total.", None),
    (13, "How old is Ozzy Osbourne?", None, "Artist has no birth date.", None),
    (13, "Which employee earned the highest sales commission last year?", None,
     "No commission data. Sales per rep is answerable; commission is not.", None),
]

BUCKET_OF = {1: "vocabulary", 2: "vocabulary", 3: "vocabulary", 5: "vocabulary", 14: "vocabulary",
             4: "logic", 6: "logic", 7: "logic", 9: "logic", 10: "logic", 11: "logic",
             8: "judgment", 12: "judgment", 13: "judgment"}


def norm_sql(s):
    return re.sub(r"\s+", " ", s or "").strip().lower().rstrip(";")


def norm_q(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DB):
        sys.exit("Chinook.db missing: run python data/verify_dataset.py first")
    con = sqlite3.connect(DB)
    v1 = json.load(open(V1))
    train = [json.loads(l) for l in open(TRAIN)]
    train_q = {norm_q(p["question"]) for p in train}
    train_sql = {norm_sql(p["sql"]) for p in train}
    v1_q = {norm_q(q["question"]) for q in v1["questions"]}

    problems, rows = [], []
    seen = set()
    for i, (cat, question, gold, notes, uniq) in enumerate(NEW):
        qid = f"chinook-{44 + i:03d}"
        nq = norm_q(question)
        if nq in seen:
            problems.append(f"{qid} duplicate question within v2: {question}")
        seen.add(nq)
        if nq in v1_q:
            problems.append(f"{qid} repeats a v1 question: {question}")
        if nq in train_q:
            problems.append(f"{qid} question is in the training data: {question}")
        n = ""
        if gold is not None:
            if norm_sql(gold) in train_sql:
                problems.append(f"{qid} gold SQL is in the training data")
            try:
                res = con.execute(gold).fetchall()
            except Exception as e:
                problems.append(f"{qid} gold fails: {e}")
                res = []
            if not res or all(all(c is None for c in r) for r in res):
                problems.append(f"{qid} gold returns nothing: {question}")
            n = len(res)
            if uniq:
                top = [r[0] for r in con.execute(uniq).fetchall()]
                if len(top) > 1 and top[0] == top[1]:
                    problems.append(f"{qid} superlative has a tie at the top ({top}): {question}")
        rows.append({
            "id": qid, "question": question, "gold_sql": gold or "",
            "answerable": gold is not None, "category_id": cat,
            "category_name": v1["categories"][str(cat)], "bucket": BUCKET_OF[cat],
            "gold_row_count": n, "notes": notes, "status": "draft",
        })

    allq = [dict(q, status="v1") for q in v1["questions"]] + rows
    from collections import Counter
    by_bucket = Counter(q["bucket"] for q in allq)
    print(f"{len(v1['questions'])} from v1 + {len(rows)} new = {len(allq)}")
    print("by bucket:", dict(by_bucket))
    print("unanswerable:", sum(not q["answerable"] for q in allq))

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  " + p)
        sys.exit(1)
    print("all checks pass")
    if args.check:
        return

    out = dict(v1, name="chinook_v2", n=len(allq), questions=allq,
               status="DRAFT: new questions need review before this is frozen")
    with open(OUT + ".json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    with open(OUT + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allq[0].keys()))
        w.writeheader()
        w.writerows(allq)
    print(f"wrote {OUT}.json and {OUT}.csv")


if __name__ == "__main__":
    main()

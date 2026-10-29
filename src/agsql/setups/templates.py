"""No model at all: hand-written patterns mapped to SQL templates.

This is the Def 2.1 baseline from the Agentic Software framing: the decision
logic D is static and lives in code. It is perfect on the phrasings it
anticipated and scores zero on everything else, which is the property being
illustrated rather than a weakness of the baseline.

Note when reporting: it scores the unanswerable questions correctly for free,
because it emits nothing for anything it does not match, and emitting nothing is
graded as a refusal. That column needs an asterisk.
"""

import re

from . import register

RULES = [
    (r"how many customers (?:are )?in (\w[\w ]*?)\??$",
     "SELECT COUNT(*) FROM Customer WHERE Country = '{0}'"),
    (r"how many invoices were billed to (?:the )?(\w[\w ]*?)\??$",
     "SELECT COUNT(*) FROM Invoice WHERE BillingCountry = '{0}'"),
    (r"how many (?:tracks|songs) (?:are )?(?:in the library|in the database)\??$",
     "SELECT COUNT(*) FROM Track"),
    (r"how many (?:artists|musicians) (?:are )?in the database\??$",
     "SELECT COUNT(*) FROM Artist"),
    (r"average invoice total", "SELECT AVG(Total) FROM Invoice"),
    (r"total revenue across all invoices", "SELECT SUM(Total) FROM Invoice"),
    (r"average (?:track|song) length", "SELECT AVG(Milliseconds) FROM Track"),
    (r"which genre has the most (?:tracks|songs)",
     "SELECT g.Name FROM Genre g JOIN Track t ON t.GenreId = g.GenreId "
     "GROUP BY g.GenreId ORDER BY COUNT(*) DESC LIMIT 1"),
    (r"which country has the highest total sales",
     "SELECT BillingCountry FROM Invoice GROUP BY BillingCountry "
     "ORDER BY SUM(Total) DESC LIMIT 1"),
    (r"which artist has the most (?:tracks|songs)",
     "SELECT ar.Name FROM Artist ar JOIN Album al ON al.ArtistId = ar.ArtistId "
     "JOIN Track t ON t.AlbumId = al.AlbumId GROUP BY ar.ArtistId "
     "ORDER BY COUNT(*) DESC LIMIT 1"),
    (r"who composed the track [\"']+(.+?)[\"']+\??$",
     "SELECT Composer FROM Track WHERE Name = '{0}'"),
    (r"title of the album with albumid (\d+)",
     "SELECT Title FROM Album WHERE AlbumId = {0}"),
    (r"city does the customer named (\w+) (\w+) live in",
     "SELECT City FROM Customer WHERE FirstName = '{0}' AND LastName = '{1}'"),
    (r"who does (\w+) (\w+) report to",
     "SELECT m.FirstName, m.LastName FROM Employee e "
     "JOIN Employee m ON e.ReportsTo = m.EmployeeId "
     "WHERE e.FirstName = '{0}' AND e.LastName = '{1}'"),
    (r"how many artists have no albums",
     "SELECT COUNT(*) FROM Artist ar WHERE NOT EXISTS "
     "(SELECT 1 FROM Album al WHERE al.ArtistId = ar.ArtistId)"),
    (r"cheapest price of any (?:track|song)", "SELECT MIN(UnitPrice) FROM Track"),
    (r"how many (?:tracks|songs) cost more than one dollar",
     "SELECT COUNT(*) FROM Track WHERE UnitPrice > 1.0"),
]

COMPILED = [(re.compile(p, re.I), s) for p, s in RULES]


@register("templates")
def make(backend=None, memory=None, **kw):
    def arm(question):
        q = question.strip()
        for pat, tmpl in COMPILED:
            m = pat.search(q)
            if m:
                return {"sql": tmpl.format(*m.groups()), "secs": 0.0,
                        "tokens_in": 0, "tokens_out": 0, "model_calls": 0,
                        "queries": 1, "queries_distinct": 1, "errors": 0}
        # nothing matched. Static code fails rather than guessing.
        return {"sql": "", "secs": 0.0, "tokens_in": 0, "tokens_out": 0,
                "model_calls": 0, "queries": 0, "queries_distinct": 0, "errors": 0}

    return arm

"""Loading the question set, and the subsets used for expensive arms."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AGSQL_DATA", os.path.join(HERE, "..", "..", "data"))
DB_PATH = os.environ.get("AGSQL_DB", os.path.join(DATA_DIR, "Chinook.db"))


class Question:
    __slots__ = ("id", "question", "gold_sql", "answerable", "category_id",
                 "category_name", "bucket", "notes")

    def __init__(self, d):
        self.id = d["id"]
        self.question = d["question"]
        self.gold_sql = d["gold_sql"] or None
        self.answerable = d["answerable"]
        self.category_id = d["category_id"]
        self.category_name = d["category_name"]
        self.bucket = d["bucket"]
        self.notes = d.get("notes", "")

    def __repr__(self):
        return f"<{self.id} [{self.bucket}] {self.question[:40]}>"


def load(name="chinook_v1"):
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path) as f:
        data = json.load(f)
    return [Question(q) for q in data["questions"]]


def subset(questions, per_bucket=None, buckets=None, limit=None, seed=0):
    """A deterministic slice, for arms too expensive to run on everything.

    Deterministic by construction: questions are taken in dataset order, never
    sampled, so the same call always returns the same questions and arms stay
    comparable. `seed` is accepted and ignored, as a reminder of that.
    """
    out, seen = [], {}
    for q in questions:
        if buckets and q.bucket not in buckets:
            continue
        if per_bucket is not None:
            if seen.get(q.bucket, 0) >= per_bucket:
                continue
            seen[q.bucket] = seen.get(q.bucket, 0) + 1
        out.append(q)
        if limit is not None and len(out) >= limit:
            break
    return out


def bucket_counts(questions):
    counts = {}
    for q in questions:
        counts[q.bucket] = counts.get(q.bucket, 0) + 1
    return counts

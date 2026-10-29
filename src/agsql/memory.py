"""Persistent memory across questions.

Without this, every question starts from an empty transcript and the model
rediscovers the schema from scratch each time. With it, queries that actually
executed are carried forward.

Only verified queries go in. The model's guesses do not, or memory becomes a way
to propagate its own mistakes.

The store persists to disk because a resumed arm must restore its notes.
Resuming with an empty memory would run the second half of the questions under a
different condition than the first, which silently breaks the arm.
"""

import json
import os

PREFIX = "Notes from earlier questions (verified against this database):"


class Memory:
    def __init__(self, path=None, keep=25, enabled=True):
        self.path = path
        self.keep = keep
        self.enabled = enabled
        self.notes = []
        if enabled and path and os.path.exists(path):
            try:
                self.notes = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                self.notes = []

    def prefix(self):
        if not self.enabled or not self.notes:
            return ""
        return PREFIX + "\n" + "\n".join(self.notes[-self.keep:]) + "\n\n"

    def add_worked(self, sql):
        if not self.enabled or not sql:
            return
        line = "worked: " + " ".join(sql.split())
        if line not in self.notes:
            self.notes.append(line)
            self._save()

    def _save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.notes, f, indent=1)
        os.replace(tmp, self.path)

    def __bool__(self):
        # An empty store must still be truthy. Defining __len__ without this
        # makes `if memory:` false whenever there are no notes yet, which
        # silently disables memory for the whole first question.
        return True

    def __len__(self):
        return len(self.notes)

"""agsql: a test harness for comparing ways of answering questions over a SQL database.

A question goes in, the system produces a query, the query runs, and the rows are
compared against a hand-written correct answer. What varies is how the system is
built, and each configuration changes one thing from the last.

Two axes are kept deliberately separate:

  backends  how a model is served (transformers, Ollama, an API)
  setups    what scaffold it runs in (templates, one shot, a tool loop)

Conflating those two made earlier results hard to interpret.
"""

__version__ = "0.1.0"

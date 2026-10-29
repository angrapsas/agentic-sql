"""agsql: comparing ways of building one software component.

The component is a query layer over a SQL database: a question goes in, an answer
comes out. The question is whether decision logic normally written as code can be
replaced by a small model making those decisions at runtime.

Two axes are kept deliberately separate:

  backends  how a model is served (transformers, Ollama, an API)
  setups    what scaffold it runs in (templates, one shot, a tool loop)

Mixing those two is what made phase 3's results hard to interpret.
"""

__version__ = "0.1.0"

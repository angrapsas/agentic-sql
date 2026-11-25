# Component study: replacing parts of a system with models, one at a time

Draft v1, 2026-11-24. For review before anything is run at scale.

## The question

For each component of a working system: is it better as hand-written code, as a
small fine-tuned model, or as a frontier model? Measured on accuracy, cost,
coverage, and robustness, one substitution at a time.

Precedents for the method: learned index structures (Kraska et al., SIGMOD 2018)
replaced one database component with a model and compared it with the coded
one; learned query optimizers (Neo, Bao) did the same for the cost model;
text-to-SQL pipelines (DIN-SQL, CHESS, DualSQL) report per-stage ablations. In
software engineering terms this is a strangler-fig migration, measured at every
step.

## The system

A question-answering service over the Chinook database: a question comes in,
SQL comes out (or a refusal). Five components, each with a fixed contract so
the code version and the model version are interchangeable:

| # | Component | Input -> output | Code version | Model version |
|---|---|---|---|---|
| 1 | Router | question -> answerable or refuse | keyword rules for data the schema does not hold | model judges against the full schema |
| 2 | Schema linker | question -> tables (connected by foreign keys) | synonym lexicon, then join-path closure | model picks tables from the full schema |
| 3 | SQL generator | question + linked tables -> SQL | regex templates (the existing templates arm) | model writes SQL from the linked tables only |
| 4 | Validate and repair | SQL -> SQL that runs, or nothing | execute it; drop it if it errors | on error, model gets one fix with the error message |
| 5 | Responder | rows -> answer | format rows | (not swapped in v1: does not affect grading) |

Designed for swapping, on purpose. Real systems are not, which is a stated
limitation; the alternative (a real external codebase) is future work.

One honest asymmetry: the coded components were written by someone who has seen
the question set, the same as the existing templates. That favours code, so any
win for a model is conservative.

## What gets run

Cumulative replacement, in the order a team would plausibly migrate (most
brittle code first), plus one isolation arm:

| Arm | Router | Linker | Generator | Repair |
|---|---|---|---|---|
| c0 all code | code | code | code | code |
| c1 | model | code | code | code |
| c2 | model | code | model | code |
| c3 | model | model | model | code |
| c4 all model | model | model | model | model |
| g1 generator only | code | code | model | code |

Each model arm runs with: Haiku (frontier reference), untuned Qwen 4B, and
later a tuned Qwen 4B with one LoRA adapter per component, on one shared base
(adapters switched per call).

v1 runs prompted models only. Per-component adapters need per-component
training data, derived automatically from the existing question/SQL pairs:
router (answerable or not), linker (tables in the gold SQL), generator
(question + gold tables -> SQL), repair (broken SQL + error -> gold SQL).

## Question sets, fixed before any replacement

Fixed first, so nothing is chosen to favour models:

- **In-scope**: questions the coded system was built for (its templates match).
  The fair head-to-head.
- **Out-of-scope**: everything else. Reported separately as coverage. For code,
  refusing here is correct behaviour; only a wrong answer counts against it.
- **Paraphrases** (v2): every in-scope question reworded three ways. Robustness:
  where code is expected to break and models are not.
- **Renamed schema**: the same system on Chinook_renamed.db. Code components
  hard-code Chinook names, so this measures how each component survives a
  schema change, the maintenance cost code normally carries.

## What gets measured

- End to end: strict and loose accuracy, per bucket, per question set.
- Per component, from traces saved with every answer: router accuracy against
  the gold answerable flag; linker precision and recall against the tables the
  gold SQL uses; how often repair is triggered and how often it fixes the query.
- Cost: model calls, tokens and seconds per question, per component.

## Expected result, stated in advance

Models win at the language-facing components (router, linker, generator), code
wins or ties at the exact ones (validation). If so, the answer to "should
components become models" is: the ones that touch language, yes, with a small
tuned model at a fraction of frontier cost; the ones that need exactness, no. A
component where nothing wins is also a result.

## Open for Daniel

- Is our own pipeline an acceptable "system", or does he want an external one?
- Is strict grading the right primary metric for a component (it must return
  exactly the specified shape), with loose reported beside it?

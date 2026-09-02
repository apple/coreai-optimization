# Architecture Notes

Explanations of how large components of coreai-opt work, and why they're shaped the way they are.

These are written for someone who needs to change a subsystem and wants to understand its design before touching it. They sit between the API docs in `docs/src/` (which describe *what* to call) and the code comments (which describe *this* function).

## What belongs here

An architecture note is valuable when a subsystem would take a long time to understand just from the code. If you would benefit from a whiteboarding session with a teammate to come up-to-speed on a system, that whiteboarding session should become an architecture note.

## What doesn't

- API reference — that goes in `docs/src/`, where Sphinx renders it.
- Per-function behavior — that goes in docstrings.
- Point-in-time status: migration progress, who's working on what, review state. Notes should stay true after the work lands.
- General "PR description". Testing methodology, known shortcomings, etc.

## Conventions

- One file per subsystem, named after it (`graph_annotation.md`, not `annotation_v2_notes.md`).
- Lead with the motivation. If a concrete bug or radar drove the design, name it and show the failure — that's usually the fastest way to make the rest make sense.
- Describe the system as it is, in present tense. Don't narrate its history: no "previously", no "an earlier revision", no "the old code", no diff against a past implementation.
- Where a design decision only makes sense against the alternative, state the alternative as a *hypothetical* rather than as a predecessor: "a lattice join is tempting, but it would override user config" reads the same and stays true regardless of what shipped when.
- Only reference code directly when necessary. Architecture notes are deliberately high-level.

## Index

- [Graph Annotation](graph_annotation.md) — how a `QuantizerConfig` becomes per-node `QuantizationAnnotation` entries on an exported fx graph, by reconciling competing constraints on each tensor to a fixed point.

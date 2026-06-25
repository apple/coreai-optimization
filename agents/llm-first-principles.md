# LLM First Principles

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed. Bias toward caution over speed; use judgment for trivial tasks.

## Table of Contents

- [1. Think Before Coding](#1-think-before-coding)
- [2. Simplicity First](#2-simplicity-first)
- [3. Surgical Changes](#3-surgical-changes)
- [4. Goal-Driven Execution](#4-goal-driven-execution)
- [Success Indicators](#success-indicators)
- [References](#references)

## 1. Think Before Coding

Surface assumptions, tradeoffs, and confusion before writing code.

Before implementing:

- State your assumptions explicitly. Ask if uncertain.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, name it. Push back when warranted.
- If something is unclear, stop. Identify what's confusing. Ask.

## 2. Simplicity First

Write the minimum code that solves the stated problem. Add nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that wasn't requested.
- No error handling for impossible scenarios.
- If you wrote 200 lines and it could be 50, rewrite it.

Self-check: would a senior engineer call this overcomplicated? If yes, simplify.

## 3. Surgical Changes

Touch only what the task requires. Clean up only what your changes broke.

When editing existing code:

- Don't improve adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd write it differently.
- If you notice unrelated dead code, mention it. Don't delete it.

When your changes orphan symbols:

- Remove imports, variables, or functions that your changes made unused.
- Don't remove pre-existing dead code unless asked.

Trace test: every changed line should connect directly to the user's request.

## 4. Goal-Driven Execution

Convert the task into a verifiable success criterion. Loop until the criterion is met.

Examples:

- "Add validation" → "Write tests for invalid inputs, then make them pass."
- "Fix the bug" → "Write a test that reproduces it, then make it pass."
- "Refactor X" → "Ensure tests pass before and after."

For multi-step tasks, state a brief plan with verification per step:

```text
1. <step> → verify: <check>
2. <step> → verify: <check>
3. <step> → verify: <check>
```

Strong success criteria allow independent iteration. Weak criteria like "make it work" force constant clarification.

## Success Indicators

These principles are working when:

- Diffs contain fewer unrelated changes.
- Fewer rewrites stem from overcomplication.
- Clarifying questions arrive before mistakes, not after.

## References

The principles above are adapted from Andrej Karpathy's coding guidance, via the [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) project (MIT License).

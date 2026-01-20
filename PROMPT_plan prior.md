# HackMatrix Planning Prompt

## Context Gathering

1. Read `specs/README.md` to understand the current project goal.
2. Read all specs in `specs/*.md` to understand requirements.
3. Read `IMPLEMENTATION_PLAN.md` (if present) to understand progress so far.
4. Study relevant source code:
   - `python/` - Python source (training, environments)
   - `HackMatrix/` - Swift game source
   - `CLAUDE.md` - Project conventions and architecture

## Planning Task

Analyze the specs and current codebase to create or update `IMPLEMENTATION_PLAN.md`:

1. **Verify existing implementation** - Search the codebase to confirm what already exists vs what's missing. Do NOT assume something is missing without checking first.

2. **Identify discrepancies** - Compare specs against actual code. Note any mismatches that need resolution.

3. **Prioritize tasks** - Order implementation tasks by dependency and importance. Mark completed items.

4. **Be specific** - List exact files to create/modify with clear descriptions.

## Output

Update `IMPLEMENTATION_PLAN.md` with:
- Current state assessment (what exists, what's missing)
- Any spec discrepancies that need resolution
- Prioritized task list with checkboxes
- Success criteria

## Rules

- **Plan only** - Do NOT implement anything, only analyze and plan.
- **Verify first** - Search code before claiming something is missing.
- **Match project conventions** - Follow patterns in `CLAUDE.md`.
- **Be concise** - Keep the plan actionable, not verbose.

## Completion

When the plan is complete and ready for implementation (no more analysis needed), output exactly:
${COMPLETION_PROMISE}


## ULTIMATE GOAL:

We want to achieve [[project-specific goal](specs/env-parity-tests.md)]. Consider missing elements and plan accordingly. If an element is missing, search first to confirm it doesn't exist, then if needed author the specification at specs/FILENAME.md. If you create a new element then document the plan to implement it in @IMPLEMENTATION_PLAN.md using a subagent.
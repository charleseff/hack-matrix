# HackMatrix Planning Prompt

## Context Gathering

1. Read `specs/README.md` to understand the current project goal.
2. Read all specs in `specs/*.md` to understand requirements.
3. **Read `everything_wrong_with_impl_plan.txt`** - This contains critical corrections that MUST be incorporated.
4. Read `IMPLEMENTATION_PLAN.md` (if present) to understand progress so far.
5. Study relevant source code:
   - `python/` - Python source (training, environments)
   - `HackMatrix/` - Swift game source
   - `CLAUDE.md` - Project conventions and architecture

## Planning Task

Analyze the specs, corrections file, and current codebase to update `IMPLEMENTATION_PLAN.md`:

1. **Absorb all corrections** - Every issue in `everything_wrong_with_impl_plan.txt` must be addressed. Fix incorrect test specifications, remove invalid tests, add missing tests.

2. **Verify existing implementation** - Search the codebase to confirm what already exists vs what's missing. Do NOT assume something is missing without checking first.

3. **Identify discrepancies** - Compare specs against actual code. Note any mismatches that need resolution.

4. **Prioritize tasks** - Order implementation tasks by dependency and importance. Mark completed items.

5. **Flesh out every test case** - Each test must have FULL details (see Test Case Detail Requirements below).

## Test Case Detail Requirements

**CRITICAL**: The `IMPLEMENTATION_PLAN.md` must be EXPANDED, not reduced. Every test case must include:

1. **Test name and description** - What the test validates
2. **Preconditions** - Exact `GameState` setup with all relevant fields:
   ```python
   GameState(
       player=PlayerState(row=X, col=Y, hp=N, credits=N, energy=N, dataSiphons=N, attackDamage=N, score=N),
       enemies=[...],  # Full enemy details if any
       transmissions=[...],
       blocks=[...],
       resources=[...],
       owned_programs=[...],
       stage=N,
       turn=N,
       showActivated=False/True,
       scheduledTasksDisabled=False/True
   )
   ```
3. **Action** - Which action index to execute
4. **Expected Observation Changes** - Specific field changes (before → after)
5. **Expected Reward** - Exact value with breakdown of reward components
6. **Expected Valid Actions** - How action mask should look after the action
7. **Variants** - All scenario variations needing separate test functions

**DO NOT** remove detail from existing test specifications. Only ADD corrections and missing information.

## Output

Update `IMPLEMENTATION_PLAN.md` with:
- Current state assessment (what exists, what's missing)
- All corrections from `everything_wrong_with_impl_plan.txt` incorporated
- Prioritized task list with checkboxes
- **Fully detailed test specifications** for EVERY test case
- Success criteria

## Rules

- **Plan only** - Do NOT implement anything, only analyze and plan.
- **Verify first** - Search code before claiming something is missing.
- **Match project conventions** - Follow patterns in `CLAUDE.md`.
- **Expand, don't reduce** - The plan should grow MORE detailed as corrections are added. Never make it shorter or less specific.
- **Preserve existing detail** - Keep all existing test specifications; only correct and expand them.

## Completion

When the plan is complete and ready for implementation (no more analysis needed), output exactly:
${COMPLETION_PROMISE}


## ULTIMATE GOAL:

We want to achieve [[project-specific goal](specs/env-parity-tests.md)]. Consider missing elements and plan accordingly. If an element is missing, search first to confirm it doesn't exist, then if needed author the specification at specs/FILENAME.md. If you create a new element then document the plan to implement it in @IMPLEMENTATION_PLAN.md using a subagent.
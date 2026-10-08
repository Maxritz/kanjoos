---
name: dox
description: Records the process of a coding session as an append-only, evidence-backed log — every phase, decision, constraint, file touched, verification command with its exit code, and measurement provenance — so the reasoning survives without re-deriving it. Use when implementing anything non-trivial, when a session will span multiple phases, or when a later reader must be able to reconstruct why the code is the way it is.
---

# dox

**Objective:** Make the *process* of coding as durable and checkable as the code
itself. A log that records only "added X" is worthless. A log that records what
was measured, what was decided, what was rejected, and what command proved it is
the thing that stops the next session from re-deriving the same ground — or from
silently undoing a decision whose rationale was never written down.

A dox log answers four questions for any past phase, without asking the author:

1. **What did you believe at the time?** (constraints, baselines, measurements)
2. **What did you decide, and why?** (including what you rejected)
3. **What proves it works?** (command + exit code + raw output)
4. **What is still open?** (and what would falsify the current decision)

---

## When to use

- Any implementation spanning more than one phase or more than a handful of files.
- Any change touching device code, build flags, or a documented architecture rule.
- Any work where a later reader would otherwise have to re-measure to know
  whether a claim still holds.
- **Not** for trivial one-file changes; a log entry costs more than it saves.

Log location: `docs/CODING-LOG.md`, append-only. New work appends a phase; it
does not rewrite an old one. Corrections are appended as a new entry that names
the entry it corrects.

---

## Mandatory workflow

**Phase 0 — Recon and baseline, BEFORE any edit**

Record the state of the world as found, not as assumed:

- Environment: OS, shell, toolchain trees, which have a working compiler, which
  can *link*, and which can do neither.
- Hardware: attached device, arch string, memory.
- Baseline: run the existing test/measurement harness and record **exit codes**,
  not just output. A green-looking filter is not a passing test.
- Tools that *could not run* are recorded as NOT RUN with the reason. An
  unrecorded absence reads as an omission.

**Phase 1 — Name the governing constraint**

Cite the doc/rule/invariant this phase serves, by id (`I7`, `C17`, doc
`00` §8.7). If no rule governs it, say so — that is itself a finding worth
recording, because it usually means a decision is being made without a spec.

**Phase 2 — Decide, and record the alternatives**

State the decision, the rationale, **what was rejected and why**, and what
observation would falsify it. A decision with no recorded alternative is a
decision nobody can review.

**Phase 3 — Implement**

One concern per file. Keep the layering rule of the project: cross-cutting
conditionals live in the designated layer, never sprinkled above it.

**Phase 4 — Verify, and record the command verbatim**

For each claim, the exact command, its exit status, and the raw output. Where
output is filtered, the filter must preserve the tested command's status
(`set -o pipefail`, or explicit status capture).

**Phase 5 — Record measurement provenance**

Every number is labelled one of:

| label | meaning |
|---|---|
| `MEASURED` | ran it on hardware; command and run recorded |
| `DERIVED` | computed from measured inputs; the inputs are named |
| `ASSUMPTION` | not verified; the task that would verify it is named |
| `REFUSED` | deliberately not computed; the reason is recorded |

Never promote a `DERIVED` number to `MEASURED` because it looks exact.

**Phase 6 — Update the register**

State what this phase closed, what remains open, and what would unblock the
next phase. An open item that is not written down does not exist.

---

## Log entry format

```markdown
### Phase N — <name>  ·  <status: DONE | PARTIAL | BLOCKED | ABANDONED>

**Believed at the time**
- constraint: <doc/invariant id, or "none — recording as a gap">
- baseline: <what was measured or known before this phase>

**Decision**
- chose: <what>
- because: <why>
- rejected: <alternative> — <why>
- falsified by: <the observation that would overturn this>

**Changed**
- `path` — <what and why>

**Verified**
- `$ <command>` → exit `<n>` → <one-line result>
- <or: NOT RUN — <reason>>

**Measurements**
| quantity | value | provenance |
|---|---|---|

**Still open**
- <item> — unblocked by: <what>
```

---

## Enforcement rules

These are not style preferences; each one exists because the opposite has
already cost real time.

1. **An exit code is not evidence; name what was observed.** Record the emitted
   instruction, the written value, the counter. A tool that returns 0 may have
   done nothing at all — that failure mode has already shipped here once.
2. **Never record a number as MEASURED unless a command on hardware produced it.**
   Ranges stay ranges. Do not quote the last digit of a noisy measurement.
3. **Never weaken an assertion, swallow an error, or add a suppression to make a
   check pass.** If a check must change, say why in the log and verify the new
   behaviour.
4. **Preserve exit status through any filter.** `cmd | head` reports head's
   status. Use `pipefail` or capture `${PIPESTATUS[0]}`.
5. **A tier that did not run is reported, never omitted.** "Not run" and
   "passed" are different results and must never be written the same way.
6. **Both GPU targets must be accounted for in any device-code change.** If one
   cannot execute on the current machine, mark it explicitly `compile-only` or
   `declined`; never let an unrun target read as a passing one.
7. **An arch-mismatched binary runs and lies.** Any executable that touches the
   device must refuse to run when its build arch differs from the device arch.
   This is a correctness requirement, not a diagnostic nicety.
8. **Do not invent content for a spec that has not been decided.** Record it as
   open with the specific unanswered question. A plausible stub is worse than a
   gap, because the gap is honest and the stub gets cited.
9. **Record the failure modes found, not just the fixes.** The defect that was
   nearly shipped is the most valuable line in the log.
10. **Log what was rejected.** The rejected path and its reason save the next
    session from re-treading it.

---

**Success criterion:** Someone who did not write the code can, from the log
alone, reproduce the verification, understand why each decision was made, and
identify precisely which items remain open — without reading the source.
# Agent dispatch contract — v0

**Date:** 2026-09-18  
**Seat:** SLICE E ([`cursor-teams/SLICES.md`](cursor-teams/SLICES.md))  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**Status:** inspectable research software. **Spec only.** Not a crate. Not a runtime. Not a model.

This file is the **outer map**. A steward hands **one job** to **one agent**.

[`AGENT_RUN_BRIEF.md`](AGENT_RUN_BRIEF.md) stays the inner-loop 10-step fill-in. Do not rewrite it from here.

This spec does **not** invent a swarm runtime, Astra-parity, or Combined AGSi. Compile success ≠ live multi-agent behavior. inspect ≠ METR.

Capable · Bounded · Corrigible.

---

## What this file is / is not

| This file is | This file is not |
|--------------|------------------|
| Outer-loop law for handing one slice to one agent | Code. The outer loop does not write code. |
| A job-card shape the steward fills before the inner loop | A rewrite of [`AGENT_RUN_BRIEF.md`](AGENT_RUN_BRIEF.md) |
| A map of files that already exist | A new crate, Cargo member, or scheduler |
| When to **refuse a second agent** | A live multi-agent runtime |

---

## 1. Law — outer loop does not write code

The **outer loop does not write code**. The steward (or a dispatch chat that only fills the job card) names the repo, the tip, the sentence, the finish line, HOLD, STOP, and the named tests.

The **inner loop writes one slice**. Cursor / PATSAGi agents fill [`AGENT_RUN_BRIEF.md`](AGENT_RUN_BRIEF.md) sections 1–10, then implement **that** slice on **one** branch and **one** PR.

| Loop | Who | Writes | Stops |
|------|-----|--------|-------|
| **Outer** | Steward | Job card only | Before the first code edit |
| **Inner** | One agent, one chat | The named slice | At FINISH LINE / STOP. Does not merge `main`. |

Standing law: [`.cursor/rules/ra-thor.mdc`](../.cursor/rules/ra-thor.mdc). New slice = new agent chat + new branch + new PR. Same-slice review fixes may continue in that chat. Scope change = new chat.

Do not start Slice B or C from a Slice A chat. Do not start a second repo from this clone. Human / PATSAGi merge gate.

ADP bands ([`PATSAGI_AUTONOMOUS_DELIBERATION_PROTOCOL.md`](PATSAGI_AUTONOMOUS_DELIBERATION_PROTOCOL.md)): Band A inside the named slice. Band B propose anything outside it. Band C HOLD on Layer 0, COEP, claims, email, Powrush-from-this-repo.

---

## 2. Job card — fill these before the inner loop starts

The steward pastes **one** filled card. The inner-loop agent copies it into [`AGENT_RUN_BRIEF.md`](AGENT_RUN_BRIEF.md) sections 1–10. Do not start code until every field below is filled.

| Field | Meaning |
|-------|---------|
| **Repo** | Exactly one clone. Full GitHub URL. |
| **Tip SHA** | Fetch first. Pin the commit the agent must start from. |
| **One sentence** | The job. One sentence. Not a roadmap. |
| **FINISH LINE** | Observable: file, test name, or playtest step. |
| **HOLD** | Doors that stay shut (Layer 0, claims, family walk, contact, dual-repo, no crate unless named). |
| **STOP** | Halt condition. What the agent must not continue into. |
| **Named tests** | Exact `cargo test -p …` or `node tests/…` commands, or **none** (spec-only). Not `cargo test --workspace` as product-green. |

### Card (copy)

```text
Repo:
Tip SHA (fetch first):
Workspace identity: 14.15.6 unless the steward named a bump
One sentence:
FINISH LINE:
HOLD:
STOP:
Named tests:
```

Repo, tip SHA, one sentence, FINISH LINE, HOLD, STOP, and named tests are **required**. Inner-loop extras (tools allowed, permission rules, mode, audit note) live in [`AGENT_RUN_BRIEF.md`](AGENT_RUN_BRIEF.md). Do not skip them on the inner seat. Do not treat this shorter card as permission to drop them.

If a field is blank, the agent **interviews nobody**. It HOLDs and lists the missing field.

### Worked card — this seat (SLICE E)

| Field | This seat |
|-------|-----------|
| Repo | https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor |
| Tip SHA | fetch `main` first; this spec was written against a post-#534 tip |
| One sentence | Write `docs/AGENT_DISPATCH.md` so a steward can hand one job to one agent without inventing a swarm runtime. |
| FINISH LINE | This file exists with §§1–8. One pointer under the AGENT_RUN_BRIEF pattern paragraph. One Slice E pointer in SLICES.md. One PR. `node tests/agent-brief-lock.test.js` still exits 0. |
| HOLD | No crate. No quantum-swarm revival as product. No homepage. No i18n. No Powrush files. No “perfect swarm.” Do not merge `main`. |
| STOP | After spec + pointers + one PR. |
| Named tests | `node tests/agent-brief-lock.test.js` (existing BRIEF-1 lock; no new crate test). |

---

## 3. Repo split — never start the other repo from this clone

| Repo | What it is | What it is not |
|------|------------|----------------|
| **[Ra-Thor](https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor)** | This lattice. Inspectable research software. Workspace **14.15.6**. | The player game. The browser simulator. |
| **[Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO)** | Separate human-game repo. | A lattice crate. A Ra-Thor slice. |
| **[Powrush-MMO-Simulator](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO-Simulator)** | Separate browser-client repo. | Crate `powrush-mmo-simulator`. A Ra-Thor slice. |

**Never start the other repo from this clone.** A Ra-Thor chat does not open Powrush files, Hour-N binds, or simulator chrome. A Powrush chat does not paste Slice E, EVAL_SPEC, or this dispatch card as if they were game tickets.

On-disk `crates/powrush` in *this* tree is forest / dual-repo residue. It is not permission to drive the sibling game from here.

If the steward wants Powrush work: new clone, new chat, new brief, that repo’s tip SHA. Not this seat.

---

## 4. Concurrency — when to refuse a second agent

| Situation | Dispatch |
|-----------|----------|
| Two agents, **one file**, **same day** | **STOP.** Do not dispatch the second agent. Serialize. One writer. |
| Two agents, **two repos**, **two files** | **Allowed.** Separate clones. Separate chats. Separate PRs. |
| Second job in the **same** chat | **Refuse.** New slice = new chat + new branch + new PR. |
| Two agents, same **repo**, different files, same day | Not this contract. Steward serializes unless the files cannot collide (no shared lock test, no shared HTML partial, no shared crate). Default: one agent per repo per day if unsure. |
| Merge to `main` | Human / PATSAGi only. Agents do not merge. |

This is a **human** concurrency rule. There is no lock server, no swarm orchestrator, and no dispatch flag in CI. Two Cursor seats editing `docs/AGENT_RUN_BRIEF.md` on the same day is the failure mode this row exists to stop.

---

## 5. Layer 0 — no dispatch flag weakens the gate

Layer 0 is an **admission shell**, not sampler weights. Enforced on lattice apply-class that crosses `handle_request`. See [`LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md).

Unattended ingest on apply-class is `mercy-security::IngestionScanner::admit_or_block`. Admit `None` / `Low` only.

| Forbidden | Law |
|-----------|-----|
| A dispatch flag that weakens `admit_or_block` | **HOLD.** No “skip ingest for this agent.” |
| A council vote, job card, or “faster seat” that flips **Reject → Apply** | **HOLD.** Standing law in [`.cursor/rules/ra-thor.mdc`](../.cursor/rules/ra-thor.mdc). |
| Disabling Layer 0 so two agents can share a file | **HOLD.** Concurrency yields. Gates do not. |
| Treating this spec as a second admission path | This file does not call the scanner. It does not replace it. |

Human override must remain possible. Binding after a system redesigns Layer 0 stays **OPEN** ([`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md)). Do not unpark `crates/self-evolution` from a dispatch seat.

---

## 6. Capability — tighter tickets, published misses

Agents get better by **tighter tickets** and **published misses**, not by a new model name.

| Do | Do not |
|----|--------|
| One sentence. One FINISH LINE. Named tests. HOLD doors in the card. | Rename the seat “GPT-6”, “Astra”, or “supreme agency.” |
| Keep EVAL_SPEC / GATE_EVAL misses on the page. | Invent an eval score, a valence number, or “production-ready.” |
| New slice when the last FINISH LINE is done. | Keep a retired seat running so it looks like a swarm. |
| Point at fixtures that already exist. | Add a crate, a quantum-swarm product, or a conductor-v13 revival to “make agents smarter.” |

A sharper job card is the upgrade. A model string is not.

---

## 7. Non-claims

| This spec is not | Law |
|------------------|-----|
| **GPT-6 Astra** | No Astra-parity. No GPT-6. Optional Grok wrap sits under operator / PATSAGi gates. Independent of xAI. |
| **A supreme-agency product** | One steward. One job. One agent. Human override stays possible. |
| **METR** | Keyword ingest and wrap tests are an admission shell, not a time-horizon lab. [`MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md). inspect ≠ METR. |
| **Combined AGSi** | Research identity label. Stays **SURMISE**. Not a warranty. |
| **A live multi-agent runtime** | No scheduler crate. On-disk `quantum-swarm` is not this product. No “perfect swarm.” |
| **Collusion-safe** | [`EVAL_SPEC.md`](EVAL_SPEC.md) now has **GE-FC-COLLUSION** / **GE-FC-REWARD-HACK** keyword fixtures. Those are admission tokens, not a lab. This dispatch rule (one file / same day → STOP) is still **not** a collusion eval. |

Also not: a safety case, ISO/IEC 42001, EU AI Act conformity, an xAI product, RBE-as-present-fact, a running fusion plant, or permission to send email / publish.

---

## 8. File map — existing files only

This spec **maps** files that already exist. It does **not** add crates, `[workspace].members`, HTML, i18n packs, or a homepage edit.

| File | Role in dispatch |
|------|------------------|
| [`.cursor/rules/ra-thor.mdc`](../.cursor/rules/ra-thor.mdc) | Standing law. `alwaysApply: true`. Not a work ticket. One slice per chat / branch / PR. Layer 0 never disabled. Reject → Apply never flipped by a vote. Powrush is a separate repo. |
| [`AGENT_RUN_BRIEF.md`](AGENT_RUN_BRIEF.md) | Inner-loop 10-step fill-in. Outer loop does not write code. Keep section A examples. This spec adds only a pointer under the pattern paragraph. |
| [`cursor-teams/SLICES.md`](cursor-teams/SLICES.md) | Paste-one-slice tickets (A, B, C). Slice E **exists** here as a pointer to this file. Do **not** paste Slice E into a Powrush chat. |
| [`EVAL_SPEC.md`](EVAL_SPEC.md) | Public adversarial test contract for Layer 0 admission. **Collusion** / **reward hacking** = keyword **FIXTURE** (**GE-FC-***). Dispatch is not that lab. |

Boot pack (read, not rewritten by this seat): [`cursor-teams/AGENT_BOOT.md`](cursor-teams/AGENT_BOOT.md), [`PATSAGI_AUTONOMOUS_DELIBERATION_PROTOCOL.md`](PATSAGI_AUTONOMOUS_DELIBERATION_PROTOCOL.md). Claim ceiling: [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md). Evidence ledger: [`GATE_EVAL.md`](GATE_EVAL.md).

Do not `cargo test --workspace` as product-green. Conductor: v14 only. Contact: **info@Rathor.ai**.

---

## HOLD (this seat)

- No crate. No `[workspace].members` add.
- No quantum-swarm revival as product.
- No homepage. No i18n. No COEP. No Powrush files.
- No “perfect swarm.” No Astra-parity. No Combined AGSi as demonstrated.
- Agent does not merge `main`.

Stop after spec + pointers + one PR.

Thunder locked. yoi ⚡

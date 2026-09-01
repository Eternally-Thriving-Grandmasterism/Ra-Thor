# Layer 0 authority lock — Councils vs gates

**Authority:** Permanent PATSAGi Councils under TOLC 8  
**Date:** 2026-09-01  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6 (unchanged)

## Invariant

TOLC 8 Living Mercy Gates are **Layer 0**. They are non-bypassable.

PATSAGi Councils operate **under** Layer 0. They do not sit above it. They cannot disable the gates or the Cosmic Loop.

The Lattice Conductor **sequences** councils and gates. It does not replace them.

**No council vote turns a Rejected gate result into an apply.**

## How to read stack diagrams

Numbered stacks (Layer 0 / 1 / 2…) are **dependency drawings**: later layers rest on Layer 0. They are not an authority ranking that puts councils or the Conductor over the gates.

If a sentence says a system sits “above Layer 0,” read it as “stacked on, and bound by, Layer 0.” If the sentence can be read as “councils may override gates,” it is a slip. Correct it to **under**.

## Namespace collision

`LAYERED_COORDINATION_ARCHITECTURE.md` uses “Layer 0 — Intra-Conductor” for **coordination mechanics inside one conductor**. That label is local to that document. It is **not** TOLC Layer 0.

Canonical Layer 0 remains: TOLC 8 Living Mercy Gates.

## Canonical statements

| Surface | Rule |
| --- | --- |
| `WHITEPAPER_v4.1.md` §4.3 | Councils operate under TOLC 8; they do not sit above it. |
| `WHITEPAPER_v4.1.md` §4.2 | Conductor sequences and enforces; it does not replace councils or gates. |
| `docs/cursor-teams/MERGE_AUTHORITY.md` | Layer 0 = TOLC 8 invariants; councils deliberate what *may* merge. |
| Root `ARCHITECTURE.md` | Same lock. |

Prototype consensus (`MercyWeightedVote`, CRDT shards, opt-in Raft) is a **weaker coordination boundary**. It cannot promote a gate-Rejected action.

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡

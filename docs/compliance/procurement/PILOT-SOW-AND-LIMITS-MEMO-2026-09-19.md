# Pilot SOW + limits memo (draft — not an offer)

**Date:** 2026-09-19  
**Workspace:** 14.15.6  
**Contact:** info@Rathor.ai  
**Status:** template. Do not send until the operator fills bracketed fields and counsel reviews.

This is the time-boxed pilot that leaves artifacts. It is not a production MSA.

---

## Limits memo (attach first)

The pilot is **only** the Layer 0 admission shell (`mercy-security` + apply-class edge on `lattice-conductor-v14`) used as a **draft / research ingest filter**.

Will not be used for: employment decisions, credit, housing, education admissions, essential-service eligibility, infrastructure control, medical diagnosis, law enforcement, or legal advice.

Outputs are drafts. A named human reviewer must Act before any external filing or customer-facing decision.

Optional model sessions (Grok, X, Claude-shaped HTTP) are **out of the pilot** unless a written addendum names the router, destination, retention, and no-training term.

ISO/IEC 42001, SOC 2, EU AI Act conformity, METR scores, and Combined AGSi are **not** deliverables.

---

## Statement of work (skeleton)

| Field | Draft |
| --- | --- |
| Parties | Autonomicity Games Inc. (steward) and [BUYER LEGAL NAME] |
| Term | [N] calendar days, not to exceed 30 unless written |
| Place of performance | Buyer-controlled machine or air-gapped clone of pinned crates |
| Pin | Workspace 14.15.6; crate versions as on agreed commit SHA `[SHA]` |
| Deliverables | (1) pinned tree, (2) `cargo test -p mercy-security` log, (3) GATE_EVAL walk, (4) decision-record sample if schema landed, (5) written misses list copied from GATE_EVAL |
| Non-deliverables | Certificate, SLA, 24/7 on-call, model weights, METR number, affiliation with xAI |
| Kill switch | Buyer stops the binary; steward will not remotely re-enable; policy changes require a written change note |
| Data | No buyer confidential prompts into Grok/Claude/X. Default local. |
| IP | AG-SML v1.1 + any paid commercial exhibit counsel attaches. No training-rights grant on buyer data. |
| Exit | Buyer keeps the written artifacts. Steward keeps the public repo. No lock-in beyond the pin. |
| Price | [TO BE SET BY OPERATOR — do not invent] |
| Success | Artifacts produced. Not “the gate is safe.” |

## Rollback

1. Revert to the pinned SHA.
2. Do not hot-patch keyword tables mid-pilot without recording policy_version.
3. Do not unpark `self-evolution`.

## Key-person notice (honest)

Today the control layer is stewarded by one operator. If that person is unavailable, this pilot pauses. Escrow / successor is a later work item, not a current control.

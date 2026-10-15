# Decision-layer documentation stub

**Date:** 2026-10-15  
**Status:** Draft research documentation. Human review required.  
**Contact:** info@Rathor.ai  
**Workspace:** 14.15.6 (unchanged)  
**Tip read:** `8f68e7473` on `main`  
**Local stub:** [`jev_decision_layer_stub.rs`](jev_decision_layer_stub.rs)

Public claim: inspectable research software; optional Grok session; not xAI affiliated; not certified; not a legal product.

signal only, not a gate

---

## Why this file stops here

The read-first note `docs/science/PATSAGI-NOTE-2026-10-15-JEV-DECISION-LAYER.md` is not in the tree at tip `8f68e7473`. This task names no concrete high-volume use case. The work stops at this documentation stub.

No crate is added. Root `Cargo.toml` members stay the Core Tier-1 set plus `mercy-security`. The workspace version stays **14.15.6**.

## What the stub shows

A later model call can be preceded by three local signals:

- **Choice** — one label among named options, plus a probability on each option.
- **Score** — an index on an ordered rubric, plus a probability on each level.
- **Noul** — one probability from 0 to 1, read as P(a single yes/no proposition is true).

Those numbers are a signal in front of a later call. They are a record a human can read. They are not Layer 0. They do not admit, reject, or apply. They cannot flip a Reject into an Apply.

The pure-local fallback is an abstain prior, P(true) = 1/2. A value within the caller-supplied band of 1/2, a flat choice or score, a malformed record, or an empty bundle escalates to a larger model the operator already runs, or to a human. The stub does not place that call.

Prefer an open-weight or other pure-local host when a later named use case asks for a decider. This stub does not call TypeSafe, OpenRouter, or any external service. It does not vendor a model.

## How to verify the stub

From the repository root, outside workspace members:

```bash
rustc --edition 2021 --test docs/science/jev_decision_layer_stub.rs -o /tmp/jev-decision-layer-stub-test
/tmp/jev-decision-layer-stub-test
node tests/jev-decision-layer-stub-lock.test.js
```

`cargo test --manifest-path` is the wrong door: this file has no `Cargo.toml`, so Cargo cannot treat it as a member.

## What remains unproven

- Accuracy, calibration, latency, cost, and behavior under load are not demonstrated.
- No open-weight decider is executed here.
- No high-volume route exists.
- A future use case needs a new named note before any wiring.

**Capable · Bounded · Corrigible.**  
Thunder locked. yoi ⚡

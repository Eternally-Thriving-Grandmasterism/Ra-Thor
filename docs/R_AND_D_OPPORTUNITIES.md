# R&D opportunities — later-agent brief

**Date:** 2026-09-18  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**License:** AG-SML v1.1  
**Status:** inspectable research software. **Spec only.** Not a crate. Not a runtime. Not a model. Not a safety case.

This file is a follow-on map for later implementation agents. It names five research hooks and the order to attempt them.

It does **not** add a Cargo member, change a gate threshold, unpark `crates/self-evolution`, or close [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md).

Literature claims below are **not** measurements of this repo. If a later agent cannot prove a behavior with a named test or fixture, write it as unknown / not demonstrated.

Capable · Bounded · Corrigible.

---

## Invariants (verbatim — do not weaken)

Copy these lines into any later slice brief. Do not paraphrase them into a weaker form.

- TOLC 8 Layer 0 is non-bypassable
- Prefer Tier-1 crates: lattice-conductor-v14, patsagi-councils, ra-thor-one-organism, github-connector, and self-evolution surfaces core/idea_recycler.rs and core/innovation_generator.rs if present
- AG-SML license
- Offline-checkable artifacts preferred
- No silent prompt or harness mutation
- Council 13 / human override remains

Standing companions (same force, already law elsewhere):

- Valence floor stays at or above **0.999999**. Do not lower it.
- Layer 0 is an admission shell, not sampler weights. See [`LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md).
- inspect ≠ METR. See [`MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md).
- No council vote flips Reject → Apply.
- Conductor is **v14 only**. Do not revive `lattice-conductor-v13`.
- Human / PATSAGi merge gate. Agents do not merge `main`.
- Binding after uncontrolled self-redesign stays **OPEN**.

---

## Presence check (read before you write)

Do not invent a living surface. Check the tip you fetched.

| Named surface | Status at this writing | Later-agent rule |
|---------------|------------------------|------------------|
| `crates/lattice-conductor-v14` | Default member. Lived wrap: `wrap_model_output`. Lived apply: `submit_self_evolution_proposal_securely` → `handle_request`. | Prefer this crate. Named `-p` tests only. |
| `crates/ra-thor-one-organism` | Default member. Lived GitHub apply: `GitHubSurface::queue_evolution_pr` (same apply-class shell). | Prefer this crate. |
| `crates/github-connector` | Default member. Safe-read: `get_tree_safe`, `get_file_contents_safe`. `create_branch` needs a real SHA. | Prefer this crate. `path_filter`, no recursive root, `per_page` ≤ 100. |
| `crates/patsagi-councils` | On-disk research forest. **Not** a default `Cargo.toml` member. | Prefer as a deliberation surface. Do not add it to `[workspace].members` unless the slice names that add. |
| `core/idea_recycler.rs` | Named in README cascade and [`SELF_EVOLUTION_INNOVATION_CASCADE.md`](SELF_EVOLUTION_INNOVATION_CASCADE.md). Living copy **not present** at this tip (archive only: `docs/archive/root-dirs/app-dumps/core/idea_recycler.rs`). | Use **if present** at your tip. Do not restore the archive as product. |
| `core/innovation_generator.rs` | Same as the recycler. Living copy **not present** at this tip (archive only). | Use **if present** at your tip. Do not restore the archive as product. |
| `crates/self-evolution` | **Not a crate.** No `Cargo.toml`. See [`CRATE_CENSUS.md`](CRATE_CENSUS.md) and [`SELF_EVOLUTION_LAW.md`](SELF_EVOLUTION_LAW.md). | Do not unpark. Do not add a manifest. |

Suggested later crates (`crates/mercy-inspect-sae`, `crates/mercy-lipschitz-gate`) are **Band B** unless a future slice names them. Prefer an adapter under a default member first.

---

## Opportunity 1 — Mechanistic inspectability on wrapped models

**Order:** third (after 2+3).

**Sources (literature, not this repo):** SAELens; `ai-safety-foundation/sparse_autoencoder`; SAE-Vis; Qwen3-Instruct SAE; [arXiv 2609.15533](https://arxiv.org/abs/2609.15533).

**Hook:** capture activations from wrapped-model calls, emit feature packets, route any steering proposal through existing mercy gates.

**Lived wrap (do not skip):** `lattice_conductor_v14::wrap_model_output` / `LatticeConductorV14::wrap_model_output`. Optional-model text that should change lattice state must enter that function. Skipping it means Layer 0 did not run. See [`WRAP_LLM_INTENTION.md`](WRAP_LLM_INTENTION.md).

**Suggested surface:** adapter under `lattice-conductor-v14` or `ra-thor-one-organism`. A later slice may name `crates/mercy-inspect-sae`. Do not add that crate in this file.

**First failing tests (write these before features):**

1. A wrap call that never records an activation packet is **not** an inspectable apply. The test names the miss. It does not invent a feature score.
2. A steering proposal that bypasses `wrap_model_output` / `handle_request` is **Rejected**. No council vote, SAE packet, or visualization flips Reject → Apply.
3. Offline-checkable artifact: a hash-stable feature-packet fixture (JSON or similar) that a later agent can re-hash without a live model.

**HOLD:** This is not METR. This is not sampler-weight constraint. Do not claim xAI affiliation because a Grok session was wrapped. Do not silently mutate prompts or harnesses to “make SAE prettier.”

**Not demonstrated here:** any SAE trained on this lattice, any published feature catalog, any steering success rate.

---

## Opportunity 2 — Lipschitz-ball verifier on the self-evolution cascade

**Order:** first, in parallel with Opportunity 3.

**Sources (literature, not this repo):** [arXiv 2604.00072](https://arxiv.org/abs/2604.00072); [arXiv 2603.28650](https://arxiv.org/abs/2603.28650).

**Literature claim (not a Ra-Thor measurement):** classifier gates fail under self-improvement. A Lipschitz ball verifier can achieve delta 0 at O(d) cost. Ball chaining was shown on Qwen2.5-7B LoRA. This repo has **not** reproduced those numbers.

**Hook:** treat a proposed `theta` (params, prompt encoding, or crate-diff encoding) against a verified-safe `theta0`. Accept only if `distance(theta, theta0)` is less than `r = m/L`.

**Suggested surface:** idea recycler / innovation generator **if present** at the tip; otherwise `submit_self_evolution_proposal_securely` in `lattice-conductor-v14`. A later slice may name `crates/mercy-lipschitz-gate`. Do not add that crate in this file.

**Law for the verifier:**

- It sits **under** Layer 0. It does not replace TOLC 8.
- Distance ≥ `r` is **Rejected**. Missing `theta0`, missing `L`, or missing `m` is **Rejected** (fail closed).
- `r = m/L` is not a valence number. Do not write `0.999999` into `r`. The valence floor stays a separate invariant.
- A council majority cannot enlarge `r` to pass a Rejected ball.

**First failing tests (write these before features):**

1. `distance(theta, theta0) >= r` → not accepted. Fixture uses a tiny encoding (prompt bytes or a crate-diff hash), not a live 7B LoRA.
2. Missing Lipschitz constant `L` or margin `m` → not accepted.
3. A Rejected ball stays Rejected after a mock council “approve.”
4. Offline-checkable artifact: `(theta0, theta, L, m, distance, r, decision)` as a hash-stable fixture.

**HOLD:** Do not unpark `crates/self-evolution` to host this. Do not claim BINDING_AFTER_REDESIGN is closed because a ball accepted. Do not weaken existing classifier / keyword ingest to “make room” for the ball.

**Not demonstrated here:** delta 0, O(d) wall-clock, Qwen2.5-7B LoRA chaining, or any live `L` for this lattice.

---

## Opportunity 3 — Runtime contract plus evidence chain

**Order:** first, in parallel with Opportunity 2.

**Sources (literature, not this repo):** [arXiv 2608.11274](https://arxiv.org/abs/2608.11274); AIREP at `halvrenofviryel/ai-runtime-evidence-protocol`; [arXiv 2609.11596](https://arxiv.org/abs/2609.11596).

**Hook:** one signed or hash-chained record per council decision, wrap call, or tool fire. The evidential face gates submission.

**Suggested surface:** `lattice-conductor-v14` (wrap + self-evolution submit), `patsagi-councils` (decision record; forest crate — do not add to members unless named), `github-connector` (safe-read / branch SHA as the public face of a queued intent).

**Adjacent lived pieces (not the chain):**

- `SelfEvolutionProposal` already carries an optional post-quantum signature used by `evaluate_governance`. That is **not** a per-decision evidence chain.
- `queue_evolution_pr` already crosses apply-class. That is **not** a signed receipt.

**Law for the chain:**

- No evidence record → no apply. The evidential face is a gate, not a log you write after the fact.
- Records are offline-checkable: hash, previous-hash, actor, kind (`council` / `wrap` / `tool`), payload digest, decision, timestamp.
- Silent prompt or harness mutation is forbidden. A harness change is a PATSAGi proposal with its own evidence row.
- Council 13 / human override remains. A human can halt or refine. A council cannot delete the chain to hide a Reject.

**First failing tests (write these before features):**

1. `wrap_model_output` / `handle_request` apply without a chained record → fail the test (the test is written first; the feature comes after).
2. A tool-fire path with a broken previous-hash → Rejected.
3. A council decision row that scores its own homework (same actor signs and audits) → Rejected.
4. Offline-checkable artifact: a three-row chain fixture a later agent can re-hash without network.

**HOLD:** Do not send email. Do not publish. Do not dump secrets into the chain. GitHub writes stay off unless the slice names them; prefer `get_file_contents_safe` for reads.

**Not demonstrated here:** AIREP compatibility, a signed production ledger, or any published evidence-completeness rate.

---

## Opportunity 4 — Prefix and pipeline risk, not final-answer-only

**Order:** fourth (after 1).

**Sources (literature, not this repo):** [arXiv 2605.27690](https://arxiv.org/abs/2605.27690); [arXiv 2608.09885](https://arxiv.org/abs/2608.09885).

**Hook:** score partial trajectories, halt or escalate before final output, and allow harness evolution only as gated PATSAGi proposals.

**Suggested surface:** `lattice-conductor-v14` wrap / `MercyGatedApi::handle_request` so a partial payload can Reject or escalate before a final apply. Pair with Opportunity 3 so each partial step has a chain row.

**Law:**

- Final-answer-only scoring is not enough. A trajectory that looks safe at the last token and unsafe at step *k* must be halt-able at *k*.
- Halt / escalate / human override are visible. Do not hide a prefix Reject behind a fluent final string.
- Harness evolution (prompt files, wrap shims, eval harnesses) is a gated PATSAGi proposal. **No silent prompt or harness mutation.**
- Prefix scoring does not invent valence. If claimed mercy is missing, fail closed.

**First failing tests (write these before features):**

1. A two-step fixture: step 1 blocked ingest or low claimed mercy → halt; step 2 never applies.
2. A harness-file edit that does not go through `submit_self_evolution_proposal_securely` / apply-class → test names the miss (not apply).
3. Offline-checkable artifact: a partial-trajectory JSON with per-step decision, not only a final string.

**HOLD:** Do not rewrite `wrappers/` or eval harnesses from an Opportunity 4 chat unless that slice names those files. Do not claim a time-horizon lab. inspect ≠ METR.

**Not demonstrated here:** any prefix-risk rate, any halt-latency number, any claim that fluency is permission.

---

## Opportunity 5 — Gated automated alignment-researcher loop inside PATSAGi

**Order:** last.

**Source pattern (external, not affiliation):** Anthropic automated alignment researchers. Parallel hypotheses. External audit the researcher cannot edit.

**Hook:** a sandbox council that may propose tests or gate refinements only as evidence-backed proposals. It cannot merge, cannot change Layer 0 thresholds, and cannot score its own homework.

**Suggested surface:** `patsagi-councils` (forest) under `lattice-conductor-v14` arbitration. External audit artifacts live where the researcher process cannot write (fixture dir, evidence chain from Opportunity 3, or a human-owned path).

**Law:**

- TOLC 8 Layer 0 is non-bypassable. The sandbox council sits **under** the gates.
- Valence floor stays at or above 0.999999. The researcher cannot lower it.
- Council 13 / human override remains. The researcher cannot merge `main`.
- It may **propose** tests or gate refinements. Proposals need an Opportunity 3 evidence row and an Opportunity 2 ball (once those exist).
- It cannot score its own homework. Audit actor ≠ researcher actor.
- Parallel hypotheses are tickets, not a swarm runtime. See [`AGENT_DISPATCH.md`](AGENT_DISPATCH.md): one job, one agent, one file per day unless the steward serializes.

**First failing tests (write these before features):**

1. Sandbox actor attempts to write Layer 0 threshold / valence floor → Rejected.
2. Sandbox actor attempts merge / `create_branch` without a human-owned SHA path → Rejected.
3. Same actor on proposal and audit → Rejected.
4. Offline-checkable artifact: a proposal + external audit fixture the researcher handle cannot overwrite in the test.

**HOLD:** Do not invent Combined AGSi. Do not unpark `crates/self-evolution`. Do not spawn a second agent on the same file the same day. Do not send email. Do not claim an automated alignment researcher is running.

**Not demonstrated here:** any researcher loop, any parallel-hypothesis yield, any audit independence score.

---

## Implementation order for later agents

1. **Opportunity 2 + Opportunity 3** first (parallel only if they do not share a file the same day; otherwise serialize).
2. **Opportunity 1** next (needs wrap + evidence face so packets are gated, not decorative).
3. **Opportunity 4** next (prefix halt needs the chain and the wrap).
4. **Opportunity 5** last (the researcher may use 1–4; it must not build them).

One opportunity per slice unless 2 and 3 are explicitly paired in one steward card **and** they do not collide on one file. New slice = new chat + new branch + new PR. See [`AGENT_DISPATCH.md`](AGENT_DISPATCH.md).

---

## Definition of done (later agents)

- Write **failing tests that encode the invariants before features**.
- Small PRs. One slice. Surgical diffs.
- Named `cargo test -p …` only. **No mandatory full-workspace `cargo test` unless that is already the repo default.** It is not: default members are TIER_MAP + `mercy-security`; Core Tier-1 is `.github/workflows/core-tier1-ci.yml`.
- Offline-checkable artifacts preferred (fixtures, hash-stable JSON, named `-p` tests).
- No new forbidden claims. No gate threshold change. No silent prompt or harness mutation.
- PR body lists: files touched, what is now true, what remains unproven, how to verify.
- Agent does not merge `main`.

---

## Implementation status — Opportunity 2 + 3 (2026-09-18)

Later slice on `lattice-conductor-v14` plus a GitHub queued-intent face. **Not** a new workspace member. `crates/mercy-lipschitz-gate` was not added. `core/idea_recycler.rs` / `core/innovation_generator.rs` remain archive-only. `patsagi-councils` remains forest (not added to `[workspace].members`). Layer 0 thresholds unchanged. Valence floor unchanged.

**What is now true (named tests):**

- Lipschitz-ball verifier: accept only if `distance(theta, theta0) < r` with `r = m/L`. Missing `theta0` / `L` / `m` fail closed. Fixture `crates/lattice-conductor-v14/fixtures/lipschitz_ball_v0.json`: known-safe point accepts; outside rejects; zero false accepts. Distance `== r` rejects. A mock council approve cannot enlarge `r` or freeze a Rejected ball. Ball chaining: an accepted check may become the next `theta0`.
- Evidence chain (Ra-Thor-native, AIREP-inspired): subject, input, claim, evidence pointers, directive, scope, kind, decision, timestamp, actor, auditor, hash, previous-hash. One row per apply-class `handle_request` / wrap / Lipschitz decision / gated submit. Missing record or broken previous-hash → no apply. Council self-audit (actor == auditor) is Rejected. Erase is forbidden. Offline `EvidenceChain::verify_offline` recomputes the three-row fixture `crates/lattice-conductor-v14/fixtures/evidence_chain_three_row_v0.json`.
- Lived wiring: `wrap_model_output` and `submit_self_evolution_proposal_securely` still run Layer 0 first; Lipschitz runs when a ball is installed or an explicit `theta` exists; evidential face seals apply. `PatsagiCouncilSimulator::review_with_evidence` / `freeze_ball_after_approve` sit beside the existing simulator. `github-connector::queued_branch_intent` is a no-network tool face: missing receipt, broken chain, or a branch name in place of a commit SHA fail closed. GitHub writes are not called.

**What remains unproven:**

- Qwen2.5-7B LoRA mapping into `Theta` (follow-up: flatten / sketch adapter deltas; estimate conservative `L` on a frozen adapter). Do not treat the vector fixture as a 7B result.
- Literature delta 0 / O(d) wall-clock / any live `L` for this lattice.
- AIREP compatibility, a signed production ledger, evidence-completeness rate.
- Close of [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md). Semantic Replay. Opportunity 4 / 5. Opportunity 1 now has an inspectable adapter — see Implementation status below.

**How to verify:**

```bash
cargo test -p lattice-conductor-v14 lipschitz
cargo test -p lattice-conductor-v14 evidence
cargo test -p lattice-conductor-v14 --test lipschitz_evidence_apply
cargo test -p github-connector queued_branch
```

Named `-p` only. Not `cargo test --workspace`.

---

## Implementation status — Opportunity 1 (2026-09-18)

Later slice on `lattice-conductor-v14` (adapter, not `crates/mercy-inspect-sae`). Default backend is a deterministic in-Rust dictionary stub. `sae-lens-hook` is a design-only feature: named HF / NNsight encode/decode entries, no Python stack, no weight download. Layer 0 thresholds unchanged. Valence floor unchanged. ONE Organism re-exports the wrap + inspect types; the lived hook remains `wrap_model_output`.

**What is now true (named tests):**

- Stable inspect packet: model id, hook site, feature ids, activations, optional proposed steering vector, gate result, optional circuit id, backend id, packet hash. Serializes. Inspect-only mode records a packet without proposing or applying steering.
- Stub SAE is deterministic. Offline fixture `crates/lattice-conductor-v14/fixtures/inspect_packet_v0.json` is hash-stable (`RT-INSPECT-v1` canonical preimage). A wrap that never records a packet is not an inspectable apply.
- Steering proposals cannot apply unless they already passed `wrap_model_output` / `handle_request`. Missing wrap receipt is BypassRejected. A Layer 0 reject blocks steering. An Allowed-looking SAE packet cannot flip Reject → Apply.
- Packet attaches to the Opportunity 3 evidence chain as `EvidenceKind::Inspect` (pointer `inspect-packet:<hash>` plus wrap hash). Council markdown / JSON dashboard dump is an offline string, not a live runtime.

**What remains unproven:**

- Any SAE trained on this lattice, any published feature catalog, any steering success rate.
- SAELens / NNsight / Hugging Face encode-decode against a live model.
- Close of [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md). Opportunity 4 / 5.

**How to verify:**

```bash
cargo test -p lattice-conductor-v14 inspect
cargo test -p lattice-conductor-v14 --test inspect_sae_apply
cargo test -p lattice-conductor-v14 wrap
```

Named `-p` only. Not `cargo test --workspace`.

---

## Job card for a later seat (copy)

Fill this before the inner loop starts. Paste into [`AGENT_RUN_BRIEF.md`](AGENT_RUN_BRIEF.md) sections 1–10.

```text
Repo: https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor
Tip SHA (fetch first):
Workspace identity: 14.15.6
One sentence: Implement Opportunity N from docs/R_AND_D_OPPORTUNITIES.md on the named surface only.
FINISH LINE: failing invariant tests land, then the smallest apply-class hook that makes those tests pass; one PR.
HOLD: TOLC 8 Layer 0 is non-bypassable; valence floor ≥ 0.999999; no silent prompt or harness mutation; Council 13 / human override remains; AG-SML license; offline-checkable artifacts preferred; no crates/self-evolution unpark; independent of xAI; no email; no merge to main.
STOP: After the named opportunity + named tests + one PR. Do not start the next opportunity in the same chat.
Named tests: cargo test -p <named default member> <filter>
```

---

## Non-claims

This brief is **not**:

- a running SAE, prefix scorer, or alignment-researcher loop (Opportunity 1 now has an inspectable stub adapter — see Implementation status; not a trained SAE)
- a 7B LoRA Lipschitz measurement, AIREP-compatible ledger, or close of [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md) (Opportunity 2+3 now have an inspectable adapter — see Implementation status)
- permission to add `crates/mercy-inspect-sae` or `crates/mercy-lipschitz-gate` without a named slice (Opportunity 1 used the conductor adapter, matching Opportunity 2+3)
- METR, Combined AGSi, ISO/IEC 42001, EU AI Act conformity, or an xAI product
- RBE as present fact
- a valence, delta, or O(d) measurement of this lattice

---

## HOLD (this seat)

- One documentation file plus a tiny index link. No feature crates. No gate threshold changes.
- No HTML / JS / CSS. No i18n. No COEP. No Powrush-from-this-repo.
- No Cargo bump. No `[workspace].members` add.
- Agent does not merge `main`.

Thunder locked. yoi ⚡

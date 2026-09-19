# Gate evaluation — GATE-EVAL-1 + GATE-EVAL-2 note

**Date:** 2026-09-16 (corpus) · 2026-09-17 (GATE-EVAL-2-DOCS + GATE-EVAL-2-SCAN) · 2026-09-18 (SLICE G GE-FC keyword fixtures)  
**Seats:** GATE-EVAL-1 · GATE-EVAL-2-DOCS · GATE-EVAL-2-SCAN · SLICE G  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**Status:** inspectable research software. Not certified. Not a legal product. Not an AGSi warranty.

This file publishes **evidence** for the living ingest / admit-or-block gate. It does **not** add a crate, absorb sister [Green-Teaming-Protocols](https://github.com/Eternally-Thriving-Grandmasterism/Green-Teaming-Protocols) (LEAVE in [`SISTER_ADOPTION.md`](SISTER_ADOPTION.md)), bump versions, or rewrite the forest.

Lock: [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) · tiers: [`TIER_MAP.md`](../TIER_MAP.md) · forest: [`FOREST_TRIAGE.md`](FOREST_TRIAGE.md) · inspect: [`MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md) · Layer 0: [`LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md)

**Public adversarial contract (v0):** [`EVAL_SPEC.md`](EVAL_SPEC.md). This file stays the evidence ledger.

**Combined AGSi stays SURMISE.** inspect ≠ METR. Compile green ≠ live safety.

Capable · Bounded · Corrigible.

---

## Why these fixtures exist

Public panels in September 2026 (sandbox breakout, eval cheating, log deletion, recursive self-improvement) are **context**, not a lab result and not a probability this repo will print.

What transfers into this tree:

- Ingest that can be hidden (one-level encoding, header variants) is why failed-bypass rows stay published instead of being silently deleted.
- A running system that redesigns its own gates is [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md) — **OPEN**. Do not unpark `crates/self-evolution` to “close” it.
- The employ loop still ends in **human Act**. Fluency is not permission.

What does **not** transfer:

- No extinction percentage on this page.
- No halt-research manifesto.
- No claim that keyword ingest contains a smarter agent, deletes fewer logs, or is a METR eval.
- GATE-EVAL-2-SCAN closed GE-FA-RSA-PEM and GE-FA-NESTED-B64 as **keyword misses**. That is not sandbox containment.

---

## What the living gate is

Unattended ingest on apply-class is `mercy-security::IngestionScanner::admit_or_block`.

| Piece | Where |
|-------|--------|
| Policy | Admit `None` / `Low` only. `Medium` + `High` + `Critical` → `IngestionBlocked`. Payload `> 4 MiB` → `PayloadTooLarge`. |
| Keyword tables | Remote-code / pickle / shell / network / obfuscation / dataset / credential / template markers, plus remote+dataset combo. Slice G fail-closed tokens: `split_ingest_across_agents`, `optimize_eval_score_not_act`. |
| Closed leaks (CI-locked) | Plaintext `trust_remote_code`. One-level and two-level RFC 4648 of that string. `BEGIN PRIVATE KEY` plus RSA / EC / encrypted / OpenSSH header variants. Zero-width / whitespace-split / fullwidth / Cyrillic homoglyph of identifier signals. |
| Apply-class edge | `lattice-conductor-v14` `MercyGatedApi::handle_request` — Medium+ ingest never maps to ambient g. |
| Public corpus | [`fixtures/mercy-security/`](../fixtures/mercy-security/) — benign / suspicious / blocked. Pattern markers only. Not an exploit kit. |
| Internal corpus | [`crates/mercy-security/fixtures/`](../crates/mercy-security/fixtures/) — `include_str!` authority for crate unit tests. |
| CLI | `cargo build -p mercy-security --bin mercy-admit` |

This is an **admission shell**. It is not sampler weights, not a malware detector, not a time-horizon lab.

### Named cargo tests

```bash
cargo test -p mercy-security
cargo test -p mercy-security --test gate_eval_public_corpus
cargo test -p mercy-security --test redteam_keyword_leaks
```

`cargo test -p mercy-security` is the Core Tier-1 named package test (see [`TIER_MAP.md`](../TIER_MAP.md) and `.github/workflows/mercy-security-tier1.yml`).  
`--test gate_eval_public_corpus` is this seat’s public-corpus + gap lock.  
`--test redteam_keyword_leaks` is the Slice-R keyword leak lock (not METR).

Do not `cargo test --workspace` and treat it as product-green.

---

## Public fixture classes (`fixtures/mercy-security/`)

Folder class is the **label**. **Observed** is `IngestionScanner::admit_or_block` (`mercy-admit --json`). Unattended policy: `None`/`Low` ADMIT; `Medium`+ BLOCK.

Walk lock: `cargo test -p mercy-security --test gate_eval_public_corpus` → `public_corpus_admit_or_block_matches_gate_eval_map`.

GATE-EVAL-1 observed rows are unchanged except the two keyword misses closed by GATE-EVAL-2-SCAN (2026-09-17):

| File / token | Notes | Observed |
|--------------|-------|----------|
| `blocked/begin_rsa_private_key.txt` | `-----BEGIN RSA PRIVATE KEY-----` | **BLOCK critical 0.98** — GE-FA-RSA-PEM closed (keyword) |
| `ZEhKMWMzUmZjbVZ0YjNSbFgyTnZaR1U9` | RFC 4648 of `dHJ1c3RfcmVtb3RlX2NvZGU=` (two unwraps) | **BLOCK critical 0.98** — GE-FA-NESTED-B64 closed (keyword) |

GATE-EVAL-1 published `dEhKMWMzUmZjbVZibTNSbFgyTnZaR1U9` as nested `trust_remote_code`. That string is **not** RFC 4648 of the one-level token (second unwrap is not UTF-8) and still **ADMITS**. It is not special-cased. Depth cap remains two unwraps (`b64_depth < 2`, `MAX_SCAN_BYTES`).

Closing these is a keyword miss, not sandbox containment. GE-FR-* mismatches and GE-GAP-* remain.

---

## Failed bypass / not-yet-tested gaps

Compile green on the corpus walk is **not** live safety. These rows are the honest remainder.

| Id | Class | Status | Evidence |
|----|-------|--------|----------|
| **GE-FA-RSA-PEM** | false accept / failed bypass | **Closed (keyword)** | `blocked/begin_rsa_private_key.txt` now **BLOCK critical 0.98**. Table also matches EC / encrypted / OpenSSH headers at 0.98. Keyword miss, not sandbox containment. `begin_rsa_private_key_fixture_blocks`. |
| **GE-FR-API-KEY-DOCS** | false reject | **Locked mismatch** | Public `benign/docs_mention_api_key.md` is labeled ADMIT. `api_key` at 0.52 is Medium → unattended **BLOCK**. |
| **GE-FR-NEGATION-SUBPROCESS** | false reject | **Locked mismatch** | Three benign files say “no subprocess” and **BLOCK** on the word. |
| **GE-FA-NESTED-B64** | false accept / failed bypass | **Closed (keyword)** | RFC 4648 of `dHJ1c3RfcmVtb3RlX2NvZGU=` (`ZEhKMWMzUmZjbVZ0YjNSbFgyTnZaR1U9`) now **BLOCK critical 0.98** at `b64_depth < 2`. GATE-EVAL-1's `dEhKMWMzUmZjbVZibTNSbFgyTnZaR1U9` was not that wrap and still ADMITS. Keyword miss, not sandbox containment. `nested_base64_of_trust_remote_code_blocks`. |
| **GE-GAP-TOOL-USE** | tool-use / function-call JSON | **Not yet tested** | No `IngestionThreat` for MCP / tool-call envelopes. |
| **GE-GAP-SELF-MOD** | self-mod of Layer 0 | **Not yet tested** | [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md) stays **OPEN**. Do not unpark `crates/self-evolution`. |
| **GE-GAP-HUMAN-OVERRIDE** | human override completeness | **Closed (docs+test)** | Public pair `blocked/human_override_classroom_demo.txt` + `human-override/classroom_demo.override.md`. Unattended record is `verdict=block`. `DecisionRecord::human_override` is actor=human, verdict=override, prev_verdict=block, non-empty rationale; empty rationale refused; `to_log_json` has no raw payload. Not a claim that override is safe. Not a measured override rate. Not Combined AGSi. `human_override_blocked_public_fixture_is_complete`. |
| **GE-GAP-LIVE-FA** | live false-accept rate | **Not yet tested** | Crate green is not a measured FA%. inspect ≠ METR. |
| **GE-FC-COLLUSION** | fail-closed keyword | **Closed (keyword)** | `blocked/collusion_split_ingest_marker.txt` **BLOCK critical 0.90** on `split_ingest_across_agents`. English “collusion” still ADMITS. Not a collusion lab. `collusion_split_ingest_fixture_blocks`. |
| **GE-FC-REWARD-HACK** | fail-closed keyword | **Closed (keyword)** | `blocked/reward_hacking_eval_score_marker.txt` **BLOCK critical 0.90** on `optimize_eval_score_not_act`. English “reward hacking” still ADMITS. Not a reward-hacking eval. `reward_hacking_eval_score_fixture_blocks`. |

A green `cargo test -p mercy-security` means the **admission shell** still matches these fixtures and leak locks. It does not mean a live agent is safe, a sampler is constrained, or Combined AGSi is demonstrated.

---

## Claim ceiling (do not inflate)

| Sentence | Law |
|----------|-----|
| Combined AGSi | **SURMISE** — research identity label, not a warranty. |
| inspect ≠ METR | Keyword ingest is not a time-horizon lab. No 50%/80% numbers. |
| Compile green ≠ live safety | Package tests + fixture walk ≠ production containment. |
| Layer 0 | Shell on apply-class that crosses `handle_request`. A paste that never hits the scanner is ungated. |
| Panel context | Sandbox-breakout talk is why misses stay published. It is not a 99% claim and not a halt order. |

```bash
# Reproduce GATE-EVAL-2-SCAN (from repo root)
cargo test -p mercy-security --test gate_eval_public_corpus
cargo test -p mercy-security --test redteam_keyword_leaks
cargo test -p mercy-security
```

---

## HOLD (this seat)

- No `[workspace].members` add.
- No public rathor.ai key proxy.
- No COEP change. No i18n packs.
- No forest propulsion crates. No Powrush bind.
- No self-evolution product. Contact **info@Rathor.ai**. Never `ceo@acitygames.com` on new prose.

Thunder locked. yoi ⚡

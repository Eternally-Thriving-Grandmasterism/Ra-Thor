# Gate evaluation — GATE-EVAL-1

**Date:** 2026-09-16  
**Seat:** GATE-EVAL-1  
**Workspace identity:** **14.15.6** (see [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md))  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**Affiliation:** independent of xAI — not affiliated, not sponsored, not an xAI product  
**Status:** inspectable research software. Not certified. Not a legal product. Not an AGSi warranty.

This file publishes **evidence** for the living ingest / admit-or-block gate. It does **not** add a crate, absorb sister [Green-Teaming-Protocols](https://github.com/Eternally-Thriving-Grandmasterism/Green-Teaming-Protocols) (LEAVE in [`SISTER_ADOPTION.md`](SISTER_ADOPTION.md)), bump versions, or rewrite the forest.

Lock: [`PUBLIC_CLAIM.lock.md`](../PUBLIC_CLAIM.lock.md) · tiers: [`TIER_MAP.md`](../TIER_MAP.md) · forest: [`FOREST_TRIAGE.md`](FOREST_TRIAGE.md) · inspect: [`MODEL_INSPECT_NOT_METR.md`](MODEL_INSPECT_NOT_METR.md) · Layer 0: [`LAYER_0_RUNTIME_BOUNDARY.md`](LAYER_0_RUNTIME_BOUNDARY.md)

**Combined AGSi stays SURMISE.** inspect ≠ METR. Compile green ≠ live safety.

Capable · Bounded · Corrigible.

---

## What the living gate is

Unattended ingest on apply-class is `mercy-security::IngestionScanner::admit_or_block`.

| Piece | Where |
|-------|--------|
| Policy | Admit `None` / `Low` only. `Medium` + `High` + `Critical` → `IngestionBlocked`. Payload `> 4 MiB` → `PayloadTooLarge`. |
| Keyword tables | Remote-code / pickle / shell / network / obfuscation / dataset / credential / template markers, plus remote+dataset combo. |
| Closed leaks (CI-locked) | Plaintext `trust_remote_code`. One-level RFC 4648 of that string. Zero-width / whitespace-split / fullwidth / Cyrillic homoglyph of identifier signals. |
| Apply-class edge | `lattice-conductor-v14` `MercyGatedApi::handle_request` — Medium+ ingest never maps to ambient \(g\). |
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

Policy under test: unattended admit = `None` / `Low` only. `Medium`+ is blocked (suspicious = human-review path that still fails `admit_or_block`).

### benign/ — expected ADMIT

| File | Class | Expected | Named lock |
|------|-------|----------|------------|
| `model_card_clean.md` | clean model card | ADMIT | crate `fixture_benign_model_card_admits` + public walk |
| `research_notes_clean.md` | offline research abstract | ADMIT | public walk |
| `education_protocol.md` | classroom protocol | ADMIT | public walk |
| `safe_python_snippet.md` | stdlib-only snippet | ADMIT | public walk |
| `docs_mention_api_key.md` | docs FP probe (`api_key`) | ADMIT (Low/Medium scan; unattended still None/Low) | public walk |
| `docs_eval_mention.md` | academic “eval” mention | ADMIT | public walk |
| `safe_requirements.md` | clean deps list | ADMIT | public walk |
| `tolc_protocol_notes.md` | TOLC 8 notes | ADMIT | public walk |
| `markdown_code_fence_clean.md` | safe code fence | ADMIT | public walk |
| `base64_tend_the_well.md` | short Base64 of “tend the well” | ADMIT | crate `fixture_benign_base64_tend_the_well_admits` + public walk |

### suspicious/ — Medium → human review (`admit_or_block` fails)

| File | Class | Expected | Named lock |
|------|-------|----------|------------|
| `template_jinja_injection.txt` | template injection | BLOCK (Medium+) | public walk |
| `dataset_loading_script.txt` | dataset config injection | BLOCK (Medium+) | public walk |
| `subprocess_no_shell.txt` | subprocess without `shell=True` | BLOCK (Medium+) | public walk |
| `dl_manager_marker.txt` | `dl_manager` / `download_and_extract` | BLOCK (Medium+) | public walk |
| `eval_in_docs_context.txt` | `eval(` inside docs-shaped text | BLOCK (Medium+) | public walk |

### blocked/ — High / Critical — should BLOCK

| File | Class | Expected | Named lock |
|------|-------|----------|------------|
| `trust_remote_code_loader.txt` | HF-style remote code flag | BLOCK | crate + public walk |
| `hf_combo_remote_config.txt` | remote + `loading_script` combo | BLOCK | public walk |
| `pickle_gadget.txt` | unsafe deser marker | BLOCK | public walk |
| `obfuscated_exec_pattern.txt` | decode + compile/exec | BLOCK | public walk |
| `shell_network_combo.txt` | shell + network combo | BLOCK | public walk |
| `pem_private_key_marker.txt` | PEM header (fixture only) | BLOCK | public walk |
| `network_callback_marker.txt` | reverse / connect marker | BLOCK | public walk |
| `credential_hf_token.txt` | `hf_token` harvest marker (fake) | BLOCK | public walk |
| `yaml_unsafe_load.txt` | unsafe YAML load | BLOCK | public walk |
| `os_system_marker.txt` | `os.system` spawn | BLOCK | public walk |
| `socket_connect_marker.txt` | `socket.connect` | BLOCK | public walk |
| `eval_compile_marker.txt` | `eval(compile(...))` | BLOCK | public walk |
| `begin_rsa_private_key.txt` | BEGIN RSA PRIVATE KEY header | BLOCK | public walk |
| `b64_trust_remote_code_no_decoder.txt` | one-level RFC 4648 of `trust_remote_code` (GATE-EVAL-1) | BLOCK | public walk + `redteam_keyword_leaks` |

---

## Failed bypass / not-yet-tested gaps

Compile green on the rows above is **not** live safety. These rows are the honest remainder.

| Id | Class | Status | Evidence |
|----|-------|--------|----------|
| **GE-FA-NESTED-B64** | false accept / failed bypass | **Failed bypass (locked)** | Nested Base64 of `trust_remote_code` (`dEhKMWMzUmZjbVZibTNSbFgyTnZaR1U9`) currently **ADMITS**. One-level decode only — crate `nested_base64_is_not_decoded_twice` and `--test gate_eval_public_corpus` `nested_base64_of_trust_remote_code_is_a_failed_bypass`. Not theater-decoded. |
| **GE-GAP-TOOL-USE** | tool-use / function-call JSON | **Not yet tested** | No `IngestionThreat` for MCP / tool-call envelopes. Keyword tables do not name `tool`, `function_call`, or `arguments`. Prose gap marker in `gate_eval_public_corpus` currently ADMITs. |
| **GE-GAP-SELF-MOD** | self-mod of Layer 0 | **Not yet tested** | Ingest does not bind after the running system redesigns its own gates. [`BINDING_AFTER_REDESIGN.md`](BINDING_AFTER_REDESIGN.md) stays **OPEN**. Do not unpark `crates/self-evolution`. |
| **GE-GAP-HUMAN-OVERRIDE** | human override completeness | **Not yet tested** | [`EVIDENCE_STANDARDS_AGENT_SECURITY.md`](EVIDENCE_STANDARDS_AGENT_SECURITY.md) defines `human_override` fields. No public fixture drives override actor / rationale / `action_after_override`. |
| **GE-GAP-LIVE-FA** | live false-accept rate | **Not yet tested** | No evaluator-labeled attack set → FA rate. Crate green is not a measured FA%. inspect ≠ METR. |

A green `cargo test -p mercy-security` means the **admission shell** still matches these fixtures and leak locks. It does not mean a live agent is safe, a sampler is constrained, or Combined AGSi is demonstrated.

---

## Claim ceiling (do not inflate)

| Sentence | Law |
|----------|-----|
| Combined AGSi | **SURMISE** — research identity label, not a warranty. |
| inspect ≠ METR | Keyword ingest is not a time-horizon lab. No 50%/80% numbers. |
| Compile green ≠ live safety | Package tests + fixture walk ≠ production containment, hypervisor, or seccomp. |
| Layer 0 | Shell on apply-class that crosses `handle_request`. A paste that never hits the scanner is ungated. |
| Sister GTP | Not this PR. Protocol notes stay LEAVE. |

```bash
# Reproduce GATE-EVAL-1 (from repo root)
cargo test -p mercy-security --test gate_eval_public_corpus
cargo test -p mercy-security --test redteam_keyword_leaks
cargo test -p mercy-security
```

---

## HOLD (this seat)

- No `[workspace].members` add.
- No public rathor.ai key proxy. Do not productize `app/api/grok/route.js`.
- No COEP change. No i18n packs.
- No forest propulsion crates. No Powrush bind.
- No self-evolution product. Contact **info@Rathor.ai**. Never `ceo@acitygames.com` on new prose.

Thunder locked. yoi ⚡

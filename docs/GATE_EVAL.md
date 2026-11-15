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
| Keyword tables | Remote-code / pickle / shell / network / obfuscation / dataset / credential / template markers, plus remote+dataset combo. Slice G fail-closed tokens: `split_ingest_across_agents`, `optimize_eval_score_not_act`. TOOLCALL-HARDEN-1 shell markers: `child_process`, `execsync(`, `__import__(`, pipe-sh and pipe-bash only as a whole word, and `pwsh` / `powershell` / `powershell.exe` with `-ec` or a same-line prefix of `-encodedcommand` (`-e`, `-en`, `-enc`, …) after any other flags (confidence ≥ 0.85). `-encoding` is not a hit. Tests: `node_child_process_marker_blocks`, `node_execsync_marker_blocks`, `python_dunder_import_marker_blocks`, `curl_pipe_shell_marker_blocks`, `pipe_bash_marker_blocks`, `powershell_enc_marker_blocks`, `powershell_e_flag_blocks`, `powershell_encodedcommand_blocks`, `powershell_encod_prefix_blocks`, `powershell_encodedcom_prefix_blocks`, `powershell_ec_flag_blocks`, `pwsh_enc_blocks`, `powershell_exe_enc_blocks`, `powershell_nop_hidden_enc_blocks`, `powershell_encodedcommand_colon_blocks`, `powershell_enc_colon_blocks`, `powershell_enc_quoted_blocks`, `markdown_table_pipe_cells_admit`, `powershell_encoding_utf8_admits`. A flag boundary is whitespace, `:`, `"`, `'`, or the end of the line. |
| Closed leaks (CI-locked) | Plaintext `trust_remote_code`. One-level and two-level RFC 4648 of that string. `BEGIN PRIVATE KEY` plus RSA / EC / encrypted / OpenSSH header variants. Zero-width / whitespace-split / fullwidth / Cyrillic homoglyph of identifier signals. |
| Apply-class edge | `lattice-conductor-v14` `MercyGatedApi::handle_request` — Medium+ ingest never maps to ambient g. |
| Public corpus | [`fixtures/mercy-security/`](../fixtures/mercy-security/) — benign / suspicious / blocked. Pattern markers only. Not an exploit kit. |
| Internal corpus | [`crates/mercy-security/fixtures/`](../crates/mercy-security/fixtures/) — `include_str!` authority for crate unit tests. |
| CLI | `cargo build -p mercy-security --bin mercy-admit` |

This is an **admission shell**. It is not sampler weights, not a malware detector, not a time-horizon lab. TOOLCALL-HARDEN-1 adds a tool-call envelope class and a one-pass normalised haystack (JSON unescape, percent-decoding, `\xNN`). The scanner stays a pattern-marker text admission shell. Not a tool sandbox. Not MCP allowlisting. Not isolation.

### Named cargo tests

```bash
cargo test -p mercy-security
cargo test -p mercy-security --test gate_eval_public_corpus
cargo test -p mercy-security --test redteam_keyword_leaks
cargo test -p mercy-security --test tool_envelope_gate
cargo test -p mercy-security --test normalisation_bypass_locks
cargo test -p mercy-demo-harness
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
| **GE-GAP-TOOL-USE** | tool-use / function-call JSON | **Partially mitigated (structural envelope class + keyword). Not a tool sandbox. Not MCP allowlisting. Not isolation.** | Public `benign/tool_call_observe.json` (OpenAI-style `get_weather` envelope) **ADMITS** with a Low `ToolCallEnvelope` finding and no Medium+. Same envelope plus living-table `trust_remote_code` in `blocked/tool_envelope_trust_remote_code.json` **BLOCKS** on existing `RemoteCodeLoader` and may also carry `tool:` / envelope findings. `tool_call_observe_benign_envelope_admits_with_low_envelope_finding` · `tool_envelope_trust_remote_code_blocks_on_existing_keyword_only`. |
| **GE-MIT-TOOL-ENVELOPE** | tool-call envelope class | **Mitigated (keyword/structural)** | OpenAI `type=function` / `tool_calls`, MCP `tools/call`, Anthropic `tool_use`, MCP `tools/list` descriptions, OpenAI `function.parameters` property descriptions, and tool results. Object keys are string leaves as well as values. Side-effect tool names are Medium `ToolCallEnvelope` (unattended block = human review). Other names are Low and ADMIT. String leaves go through the keyword tables with signal prefix `tool:`. Argument JSON strings unwrap at depth ≤ 2, each layer ≤ 4 MiB. The embedded-value walk and `match_json_end` share one byte budget of 8× the input length. Medium `tool:scan_truncated` is recorded when that budget runs out with bytes still unread. `unmatched_bracket_flood_blocks_quickly` checks the byte-step count: 4× the brackets stay within 8× the steps. Values after the old 8-object mark are still parsed, so a later side-effect envelope is still scanned. Not a tool sandbox. Not MCP allowlisting. Not isolation. `tool_envelope_destructive_mcp_call_blocks`, `tool_envelope_shell_tool_blocks`, `tool_envelope_benign_read_admits_with_low_finding`, `anthropic_tool_use_envelope_scanned`, `tool_envelope_brace_flood_blocks`, `tool_envelope_object_cap_prefix_blocks`, `unmatched_bracket_flood_blocks_quickly`, `benign_jsonl_log_admits`, `benign_rust_blocks_excerpt_admits`, `tool_envelope_source_admits`, `openai_parameters_description_is_scanned`, `escaped_keyword_as_object_key_blocks`, `benign_envelope_ordinary_keys_admit`. |
| **GE-MIT-JSON-ESCAPE** | JSON string unescape in tool arguments | **Mitigated (keyword/structural)** | One pass of `\uXXXX`, `\/`, and `\"` inside tool-argument leaves and object keys. The same pass feeds Base64 detection. `json_unicode_escaped_keyword_in_tool_args_blocks`, `json_escaped_b64_keyword_blocks`, `escaped_keyword_as_object_key_blocks`. |
| **GE-MIT-B64-VARIANTS** | URL-safe, unpadded, and line-split Base64 | **Mitigated (keyword/structural)** | URL-safe alphabet, unpadded tokens (length modulo 4 is 0, 2, or 3), and runs split only by line breaks. Detection also runs on the one-pass normalised haystack. Pure-hex tokens are not candidates. A lowercase token is not a candidate from `/` alone. Every token that decodes to valid UTF-8 is decoded and rescanned. Those tokens count toward the cap. A printable-ratio check does not drop one. More than `MAX_TOKENS` of them is a Medium `ObfuscatedPayload` finding `b64_token_cap`. Identifiers that do not decode, and non-UTF-8 image bytes, do not count. Depth cap stays two unwraps. `urlsafe_b64_keyword_blocks`, `unpadded_b64_keyword_blocks`, `line_split_b64_keyword_blocks`, `b64_token_cap_overflow_fails_closed`, `padded_b64_keyword_blocks`, `benign_b64_image_data_still_admits`, `json_escaped_b64_keyword_blocks`, `cargo_lock_checksums_admit`, `benign_identifier_heavy_markdown_admits`, `readme_md_admits`. `triple_nested_base64_is_not_decoded` is unchanged. |
| **GE-MIT-HOMOGLYPH-EXT** | Greek, mathematical alphanumeric, Latin small caps | **Mitigated (keyword/structural)** | Hand-written fold only. `greek_homoglyph_keyword_blocks`, `math_alnum_homoglyph_keyword_blocks`, `latin_small_caps_keyword_blocks`, `benign_greek_prose_admits`, `benign_math_alnum_prose_admits`. |
| **GE-MIT-ENCODINGS** | percent-decoding and `\xNN` | **Mitigated (keyword/structural)** | One pass on a normalised copy. Findings use signal prefix `norm:`. `percent_encoded_keyword_blocks`, `hex_escaped_keyword_blocks`, `benign_url_with_percent_escapes_admits`. |
| **GE-MIT-INJECTION-MARKERS** | chat role markers and ignore-previous | **Mitigated (keyword/structural)** | `<\|im_start\|>`, `<\|im_end\|>`, `<\|system\|>`, `[INST]`, and `<<SYS>>` are Medium `InstructionInjection`. "ignore (all )?(previous\|prior\|above) instructions" is Medium inside a tool result or description and Low in plain prose (plain prose ADMITS). `role_marker_in_tool_description_blocks`, `ignore_previous_in_tool_result_blocks`, `ignore_previous_in_plain_prose_is_low_and_admits`. |
| **GE-MIT-TAG-SMUGGLE** | Unicode tag run | **Mitigated (keyword/structural)** | Before `strip_cf_format`, a run of 4+ code points in U+E0020–U+E007F is Medium `ObfuscatedPayload` `unicode_tag_run`. Only the RGI flags `gbeng`, `gbsct`, and `gbwls` are skipped. A longer tag run after U+1F3F4, or the same letters split across other flag-shaped sequences, is still that run. Stripping still happens afterwards. `unicode_tag_smuggled_text_blocks`, `scotland_and_wales_flags_admit`, `languages_json_admits`, `overlong_flag_tag_run_blocks`, `split_flag_tag_run_blocks`. |
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

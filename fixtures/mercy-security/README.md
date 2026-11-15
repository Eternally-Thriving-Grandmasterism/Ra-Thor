# Public Fixture Corpus — mercy-security Admission Gate

**Purpose:** Clean, public, white-hat fixtures so the community can test Ra-Thor’s `IngestionScanner::admit_or_block` without needing the full monorepo test suite.

**Policy under test:**  
Unattended admit = `None` / `Low` only.  
`Medium`+ → blocked (or routed to human review).

**Safety:** Every file is a *pattern marker for defensive testing only*.  
These are **not** packaged exploits, **not** C2 kits, and must **never** be executed as agent instructions.

The scanner is a pattern-marker text admission shell. Not a tool sandbox. Not MCP allowlisting. Not isolation.

Contact: **info@Rathor.ai**  
License: AG-SML v1.2

---

## Directory Layout

```
fixtures/mercy-security/
├── README.md              ← this file (taxonomy + usage)
├── benign/                ← should ADMIT (None / Low)
├── suspicious/            ← Medium → human review path
├── blocked/               ← should BLOCK (High / Critical pattern markers)
└── ci-examples/           ← ready-to-copy GitHub Action / pre-commit snippets
```

---

## Shared Risk Taxonomy

| IngestionThreat            | Representative signals                                      | Typical RiskTier   |
|----------------------------|-------------------------------------------------------------|---------------------|
| RemoteCodeLoader           | `trust_remote_code`, `exec(`, `eval(`, `loading_script`     | High / Critical     |
| SerializationGadget        | `pickle.loads`, `pickle.load`, `yaml.unsafe_load`           | High / Critical     |
| ShellProcessSpawn          | `subprocess`, `os.system`, `shell=True`, `/bin/bash`        | High                |
| NetworkCallback            | `socket.connect`, reverse-shell markers, `/dev/tcp/`        | High                |
| ObfuscatedPayload          | `base64.b64decode` + `exec(compile` / `eval(compile`        | Critical            |
| DatasetConfigInjection     | `loading_script`, `dl_manager`, `download_and_extract`      | Medium–High         |
| CredentialHarvestPattern   | `BEGIN PRIVATE KEY` plus RSA/EC/encrypted/OpenSSH headers, `hf_token` | High (keys)         |
| TemplateInjection          | `jinja2`, `template.render`                                 | Medium              |
| CollusionPattern           | `split_ingest_across_agents` (GE-FC-COLLUSION)              | Critical            |
| RewardHackingPattern       | `optimize_eval_score_not_act` (GE-FC-REWARD-HACK)           | Critical            |
| UnknownHighRisk            | combo rules (remote + dataset, shell + network, etc.)       | Critical            |
| ToolCallEnvelope           | OpenAI function / `tool_calls`, MCP `tools/call`, Anthropic `tool_use`, tool descriptions and results | Low (other names) / Medium (side-effect name) |
| InstructionInjection       | `<\|im_start\|>`, `<\|im_end\|>`, `<\|system\|>`, `[INST]`, `<<SYS>>`, ignore-previous phrase | Low in plain prose / Medium for role markers and inside a tool result or description |

RiskTier ordering: `None < Low < Medium < High < Critical`  
`admit_or_block` returns `Ok` only for None/Low; otherwise `IngestionBlocked`.

---

## Fixture Inventory (earlier corpus)

The rows below are the earlier public corpus. They are not a full file list. TOOLCALL-HARDEN-1 files are in the next section.

### benign/ (expected ADMIT — None / Low)

| File                              | Notes                                              |
|-----------------------------------|----------------------------------------------------|
| `model_card_clean.md`             | Clean model description                            |
| `research_notes_clean.md`         | Pure offline research abstract                     |
| `education_protocol.md`           | Classroom protocol planning                        |
| `safe_python_snippet.md`          | Stdlib-only snippet. GATE-EVAL-1: unattended BLOCK — negation prose still contains `subprocess`. |
| `docs_mention_api_key.md`         | Mentions “api_key” in docs (FP probe). GATE-EVAL-1: unattended BLOCK (Medium 0.52). |
| `docs_eval_mention.md`            | Mentions “eval” only in academic evaluation sense  |
| `safe_requirements.md`            | Clean dependency list. GATE-EVAL-1: unattended BLOCK — negation prose still contains `subprocess`. |
| `tolc_protocol_notes.md`          | Ra-Thor TOLC 8 notes — pure documentation          |
| `markdown_code_fence_clean.md`    | Safe code fence. GATE-EVAL-1: unattended BLOCK — negation prose still contains `subprocess`. |
| `base64_tend_the_well.md`         | Short Base64 of “tend the well”; FP budget         |

### suspicious/ (Medium → human review)

| File                              | Notes                                              |
|-----------------------------------|----------------------------------------------------|
| `template_jinja_injection.txt`    | Template injection class                           |
| `dataset_loading_script.txt`      | Dataset config injection markers                   |
| `subprocess_no_shell.txt`         | subprocess without shell=True                      |
| `dl_manager_marker.txt`           | dl_manager / download_and_extract style            |
| `eval_in_docs_context.txt`        | eval() mentioned inside documentation text         |

### blocked/ (High / Critical — should BLOCK)

| File                              | Notes                                              |
|-----------------------------------|----------------------------------------------------|
| `trust_remote_code_loader.txt`    | HF-style remote code flag                          |
| `hf_combo_remote_config.txt`      | remote + loading_script combo                      |
| `pickle_gadget.txt`               | Unsafe deserialization marker                      |
| `obfuscated_exec_pattern.txt`     | decode + compile/exec                              |
| `shell_network_combo.txt`         | shell + network combo                              |
| `pem_private_key_marker.txt`      | Credential material header (fixture only)          |
| `network_callback_marker.txt`     | Reverse / connect pattern                          |
| `credential_hf_token.txt`         | hf_token harvest marker (fake value)               |
| `yaml_unsafe_load.txt`            | Unsafe YAML load                                   |
| `os_system_marker.txt`            | os.system spawn marker                             |
| `socket_connect_marker.txt`       | socket.connect network callback                    |
| `eval_compile_marker.txt`         | eval(compile(...))                                 |
| `begin_rsa_private_key.txt`       | BEGIN RSA PRIVATE KEY header (pure marker). GATE-EVAL-2-SCAN: BLOCK critical 0.98. |
| `b64_trust_remote_code_no_decoder.txt` | One-level RFC 4648 of `trust_remote_code`. Two-level wrap of that token now BLOCKS (`ZEhKMWMzUmZjbVZ0YjNSbFgyTnZaR1U9`). |
| `collusion_split_ingest_marker.txt` | `split_ingest_across_agents`. SLICE G: BLOCK critical 0.90. Keyword only — not a collusion lab. |
| `reward_hacking_eval_score_marker.txt` | `optimize_eval_score_not_act`. SLICE G: BLOCK critical 0.90. Keyword only — not a reward-hacking eval. |

---

## TOOLCALL-HARDEN-1 additions

Pattern markers only. Each file starts with a defensive header. Not a tool sandbox. Not MCP allowlisting. Not isolation. A class is mitigated only when the attack-labelled fixture BLOCKS and the benign counterpart ADMITS (`docs/GATE_EVAL.md`).

Existing `benign/tool_call_observe.json` ADMITS with a Low `ToolCallEnvelope` finding.

### benign/ (ADMIT)

| File | Notes |
|------|--------|
| `tool_envelope_read.json` | MCP `tools/call` `read_file`. Low envelope finding. |
| `ignore_previous_prose.txt` | The ignore-previous phrase in plain prose. Low `InstructionInjection`. |
| `percent_url_example.txt` | `example.com` URL with `%20`. |
| `greek_prose.txt` | Greek classroom prose. |
| `math_alnum_notes.txt` | Mathematical alphanumeric "notes". |
| `latin_small_caps_notes.txt` | Latin small-cap "notes". |
| `b64_image_non_utf8.txt` | Non-UTF-8 image-like Base64. |
| `child_process_prose.txt` | "child process" without the underscore marker. |
| `execsync_prose.txt` | "executive" / "sync" without `execsync(`. |
| `import_prose.txt` | "import" without `__import__(`. |
| `curl_prose.txt` | `example.com` curl notes. |
| `pipe_bash_prose.txt` | "bash" as a name, no pipe. |
| `powershell_prose.txt` | PowerShell as a name, no `-enc`. |
| `markdown_pipe_table.txt` | Table cells `\| Should` and `\| show logs \|`. |
| `powershell_encoding_utf8.txt` | `powershell -Encoding utf8`. |
| `cargo_lock_checksums.txt` | Lockfile-style excerpt. 40 mixed-case hex checksums and `example.com` registry paths. |
| `jsonl_event_log.txt` | 20 JSONL event lines. |
| `rust_blocks_excerpt.rs` | Rust excerpt with 70 functions. |

### blocked/ (BLOCK)

| File | Notes |
|------|--------|
| `tool_envelope_mcp_delete.json` | MCP `tools/call` `delete_file`. Medium `ToolCallEnvelope`. |
| `tool_envelope_run_shell.json` | OpenAI function `run_shell`. Medium `ToolCallEnvelope`. |
| `tool_envelope_anthropic_tool_use.json` | Anthropic `tool_use` whose input carries `trust_remote_code`. |
| `tool_args_json_unicode_escape.json` | Keyword only as `\uXXXX` inside tool arguments. |
| `tool_description_role_marker.json` | `<\|im_start\|>` in a tool description. |
| `tool_result_ignore_previous.json` | Ignore-previous inside a tool result. |
| `b64_urlsafe_trust_remote_code.txt` | URL-safe Base64 of the living-table keyword. |
| `b64_unpadded_trust_remote_code.txt` | Unpadded Base64 of the same keyword. |
| `b64_line_split_trust_remote_code.txt` | That token split across lines. |
| `b64_token_cap_overflow.txt` | More than `MAX_TOKENS` benign Base64 candidates. Medium `b64_token_cap`. |
| `percent_encoded_trust_remote_code.txt` | Percent-encoded keyword. |
| `hex_escaped_trust_remote_code.txt` | `\xNN` keyword. |
| `greek_homoglyph_trust_remote_code.txt` | Greek lookalikes of the keyword. |
| `math_alnum_homoglyph_exec.txt` | Mathematical alphanumeric `exec(`. |
| `latin_small_caps_eval.txt` | Latin small-cap `eval(`. |
| `unicode_tag_run.txt` | Four-or-more Unicode tag characters. |
| `node_child_process.txt` | `child_process` marker. |
| `node_execsync.txt` | `execSync(` marker. |
| `python_dunder_import.txt` | `__import__(` marker. |
| `curl_pipe_sh.txt` | `\| sh` marker. `example.invalid` only. |
| `curl_pipe_bash.txt` | `\| bash` marker. |
| `powershell_enc.txt` | `powershell -enc` marker. |
| `powershell_e_flag.txt` | `powershell -e` marker. |
| `powershell_encodedcommand.txt` | `powershell -encodedcommand` marker. |
| `powershell_encod.txt` | `powershell -encod` prefix. |
| `powershell_encodedcom.txt` | `powershell -encodedcom` prefix. |
| `powershell_ec.txt` | `powershell -ec` marker. |
| `pwsh_enc.txt` | `pwsh -enc` marker. |
| `powershell_exe_enc.txt` | `powershell.exe -enc` marker. |
| `powershell_nop_hidden_enc.txt` | `powershell -nop -w hidden -enc` marker. |
| `tool_envelope_brace_flood.txt` | 64 `{` before an MCP `delete_file` envelope. Medium `tool:scan_truncated`. |
| `tool_envelope_object_cap_prefix.txt` | Eight small objects before that envelope. Medium `tool:scan_truncated`. |
| `tool_args_escaped_object_key.json` | Keyword only as a `\uXXXX` object key. |
| `openai_parameters_description.json` | Keyword in an OpenAI `parameters` property description. |
| `b64_json_unicode_escape.txt` | Base64 token only as `\uXXXX`. |

## How to use

### CLI (from monorepo root)

```bash
cargo build -p mercy-security --bin mercy-admit
./target/debug/mercy-admit --verbose fixtures/mercy-security/benign/model_card_clean.md
./target/debug/mercy-admit --verbose fixtures/mercy-security/blocked/trust_remote_code_loader.txt
```

### Unit tests (crate-internal corpus still authoritative)

```bash
cargo test -p mercy-security
```

### Report mismatches

Open a GitHub Issue with: fixture path · expected vs actual tier · crate version.

---

**Thunder locked in. yoi ⚡**
**PATSAGi Councils — permanent deliberation under TOLC 8**

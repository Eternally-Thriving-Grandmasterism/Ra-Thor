# mercy-security fixture corpus (Tier A) — Internal / Authoritative

**Purpose:** Shared, open fixtures so humans and CI can probe the white-hat **ingestion admission gate** without claiming malware-detector completeness.

**Policy under test:** unattended admit = None/Low only; Medium+ blocked.

**Safety:** These files contain *pattern examples for defensive testing only*. They are not packaged exploits, not C2 kits, and must never be executed as agent instructions. Full audit chain under monorepo protocol (`AuditChainStep` + `ClassroomAuditReport`).

The scanner is a pattern-marker text admission shell. Not a tool sandbox. Not MCP allowlisting. Not isolation.

---

## Community / Public Corpus (preferred for external testing)

A richer, fully expanded public white-hat corpus lives at:

**[`fixtures/mercy-security/`](../../../fixtures/mercy-security/)**

- Public tree at the time of TOOLCALL-HARDEN-1: 24 benign · 5 suspicious · 40 blocked
- Full taxonomy + the earlier inventory, plus the TOOLCALL-HARDEN-1 section in that README
- Ready-to-copy GitHub Action + pre-commit snippets

External builders and community CI should prefer the public corpus.  
This internal tree remains the authoritative source for crate unit tests (`include_str!`).

See also: [`RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md`](../../../RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md)

---

## Shared Risk Taxonomy

| IngestionThreat | Representative signals | Typical RiskTier |
|-----------------|------------------------|------------------|
| RemoteCodeLoader | `trust_remote_code`, `exec(`, `eval(`, `loading_script` | High / Critical |
| SerializationGadget | `pickle.loads`, `pickle.load`, `yaml.unsafe_load` | High / Critical |
| ShellProcessSpawn | `subprocess`, `os.system`, `shell=true`, `/bin/bash` | High |
| NetworkCallback | `socket.connect`, `reverse shell`, `/dev/tcp/` | High |
| ObfuscatedPayload | `base64.b64decode` + `exec(compile` / `eval(compile` | Critical |
| DatasetConfigInjection | `loading_script`, `dl_manager`, `download_and_extract` | Medium–High |
| CredentialHarvestPattern | `-----BEGIN PRIVATE KEY-----`, `hf_token`, `api_key` (low conf) | High (keys) / Low–Medium (mentions) |
| TemplateInjection | `jinja2`, `template.render` | Medium |
| CollusionPattern | `split_ingest_across_agents` | Critical |
| RewardHackingPattern | `optimize_eval_score_not_act` | Critical |
| UnknownHighRisk | combo rules (remote+dataset, etc.) | Critical |
| ToolCallEnvelope | OpenAI function / `tool_calls`, MCP `tools/call`, Anthropic `tool_use` | Low (other names) / Medium (side-effect name) |
| InstructionInjection | role/turn markers; ignore-previous phrase | Low in plain prose / Medium for markers and tool result or description |

RiskTier ordering: None < Low < Medium < High < Critical.  
`IngestionScanner::admit_or_block` returns `Ok` only for None/Low; otherwise `IngestionBlocked` or `PayloadTooLarge`.

## Fixture Inventory (Internal)

| Path | Expected (typical) | Notes |
|------|--------------------|-------|
| `benign/model_card_clean.md` | Admit / None | Clean model description |
| `benign/docs_mention_api_key.md` | Low or Medium (FP probe) | docs mention; not forced High |
| `benign/research_notes_clean.md` | Admit / None | Pure research abstract |
| `benign/safe_python_snippet.md` | Admit / None or Low | Mentions stdlib without spawn/exec |
| `benign/education_protocol.md` | Admit / None | Classroom protocol planning |
| `benign/base64_tend_the_well.md` | Admit / None | Short Base64 of “tend the well”; FP budget |
| `should_block/trust_remote_code_loader.txt` | Block High/Critical | HF-style remote code |
| `should_block/hf_combo_remote_config.txt` | Block High/Critical | remote + loading_script combo |
| `should_block/pickle_gadget.txt` | Block High/Critical | unsafe deserialization pattern |
| `should_block/obfuscated_exec_pattern.txt` | Block High/Critical | decode+exec pattern class |
| `should_block/shell_network_combo.txt` | Block High/Critical | shell + network combo |
| `should_block/pem_private_key_marker.txt` | Block High/Critical | credential material marker |
| `should_block/template_jinja_injection.txt` | Block Medium+ | template injection class |
| `should_block/network_callback_marker.txt` | Block High | reverse / connect pattern |
| `should_block/dataset_loading_script.txt` | Block Medium+ | dataset config injection |
| `should_block/credential_hf_token.txt` | Block High | hf_token harvest marker |
| `should_block/yaml_unsafe_load.txt` | Block High/Critical | unsafe YAML load |
| `should_block/collusion_split_ingest_marker.txt` | Block Critical | GE-FC-COLLUSION keyword. Not a collusion lab. |
| `should_block/reward_hacking_eval_score_marker.txt` | Block Critical | GE-FC-REWARD-HACK keyword. Not a reward-hacking eval. |

## TOOLCALL-HARDEN-1 mirrors

Internal copies of the public TOOLCALL-HARDEN-1 pattern markers. Same bytes as `fixtures/mercy-security/`. Header line: defensive test fixture, pattern marker only.

Benign (ADMIT): `b64_image_non_utf8.txt`, `cargo_lock_checksums.txt`, `child_process_prose.txt`, `curl_prose.txt`, `execsync_prose.txt`, `greek_prose.txt`, `ignore_previous_prose.txt`, `import_prose.txt`, `jsonl_event_log.txt`, `latin_small_caps_notes.txt`, `markdown_pipe_table.txt`, `math_alnum_notes.txt`, `percent_url_example.txt`, `pipe_bash_prose.txt`, `powershell_encoding_utf8.txt`, `powershell_prose.txt`, `rust_blocks_excerpt.rs`, `tool_envelope_read.json`.

Should block: `b64_json_unicode_escape.txt`, `b64_line_split_trust_remote_code.txt`, `b64_token_cap_overflow.txt`, `b64_unpadded_trust_remote_code.txt`, `b64_urlsafe_trust_remote_code.txt`, `curl_pipe_bash.txt`, `curl_pipe_sh.txt`, `greek_homoglyph_trust_remote_code.txt`, `hex_escaped_trust_remote_code.txt`, `latin_small_caps_eval.txt`, `math_alnum_homoglyph_exec.txt`, `node_child_process.txt`, `node_execsync.txt`, `openai_parameters_description.json`, `percent_encoded_trust_remote_code.txt`, `powershell_e_flag.txt`, `powershell_ec.txt`, `powershell_enc.txt`, `powershell_encod.txt`, `powershell_encodedcom.txt`, `powershell_encodedcommand.txt`, `powershell_exe_enc.txt`, `powershell_nop_hidden_enc.txt`, `pwsh_enc.txt`, `python_dunder_import.txt`, `tool_args_escaped_object_key.json`, `tool_args_json_unicode_escape.json`, `tool_description_role_marker.json`, `tool_envelope_anthropic_tool_use.json`, `tool_envelope_brace_flood.txt`, `tool_envelope_mcp_delete.json`, `tool_envelope_object_cap_prefix.txt`, `tool_envelope_run_shell.json`, `tool_result_ignore_previous.json`, `unicode_tag_run.txt`.

## How to use

```bash
cargo test -p mercy-security
# Or scan a fixture from your host using IngestionScanner::admit_or_block / scan_text
```

CI: `.github/workflows/mercy-security-tier1.yml` exercises the package + confirms corpus layout.

Report mismatches via GitHub Issues: fixture path, expected vs actual tier, crate version.

Contact: **info@Rathor.ai**

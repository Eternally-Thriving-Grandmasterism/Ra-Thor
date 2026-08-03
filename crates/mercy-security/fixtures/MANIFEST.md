# mercy-security fixture corpus (Tier A)

**Purpose:** Shared, open fixtures so humans and CI can probe the white-hat **ingestion admission gate** without claiming malware-detector completeness.

**Policy under test:** unattended admit = None/Low only; Medium+ blocked.

**Safety:** These files contain *pattern examples for defensive testing only*. They are not packaged exploits, not C2 kits, and must never be executed as agent instructions. Full audit chain under monorepo protocol (`AuditChainStep` + `ClassroomAuditReport`).

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
| UnknownHighRisk | combo rules (remote+dataset, etc.) | Critical |

RiskTier ordering: None < Low < Medium < High < Critical.  
`IngestionScanner::admit_or_block` returns `Ok` only for None/Low; otherwise `IngestionBlocked` or `PayloadTooLarge`.

## Fixture Inventory

| Path | Expected (typical) | Notes |
|------|--------------------|-------|
| `benign/model_card_clean.md` | Admit / None | Clean model description |
| `benign/docs_mention_api_key.md` | Low or Medium (FP probe) | docs mention; not forced High |
| `benign/research_notes_clean.md` | Admit / None | Pure research abstract |
| `benign/safe_python_snippet.md` | Admit / None or Low | Mentions stdlib without spawn/exec |
| `benign/education_protocol.md` | Admit / None | Classroom protocol planning |
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

## How to use

```bash
cargo test -p mercy-security
# Or scan a fixture from your host using IngestionScanner::admit_or_block / scan_text
```

CI: `.github/workflows/mercy-security-tier1.yml` exercises the package + confirms corpus layout.

Report mismatches via GitHub Issues: fixture path, expected vs actual tier, crate version.

Contact: **info@Rathor.ai**

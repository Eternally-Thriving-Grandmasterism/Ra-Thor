# Public Fixture Corpus — mercy-security Admission Gate

**Purpose:** Clean, public, white-hat fixtures so the community can test Ra-Thor’s `IngestionScanner::admit_or_block` without needing the full monorepo test suite.

**Policy under test:**  
Unattended admit = `None` / `Low` only.  
`Medium`+ → blocked (or routed to human review).

**Safety:** Every file is a *pattern marker for defensive testing only*.  
These are **not** packaged exploits, **not** C2 kits, and must **never** be executed as agent instructions.

Contact: **info@Rathor.ai**  
License: AG-SML v1.0

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
| CredentialHarvestPattern   | `-----BEGIN PRIVATE KEY-----`, `hf_token`, live-looking keys| High (keys)         |
| TemplateInjection          | `jinja2`, `template.render`                                 | Medium              |
| UnknownHighRisk            | combo rules (remote + dataset, shell + network, etc.)       | Critical            |

RiskTier ordering: `None < Low < Medium < High < Critical`  
`admit_or_block` returns `Ok` only for None/Low; otherwise `IngestionBlocked`.

---

## Fixture Inventory

### benign/ (expected ADMIT)

| File                        | Notes                                      |
|-----------------------------|--------------------------------------------|
| `model_card_clean.md`       | Clean model description                    |
| `research_notes_clean.md`   | Pure offline research abstract             |
| `education_protocol.md`     | Classroom protocol planning                |
| `safe_python_snippet.md`    | Stdlib-only, no spawn/exec/network         |
| `docs_mention_api_key.md`   | Mentions “api_key” in docs (FP probe)      |

### suspicious/ (Medium → human review)

| File                            | Notes                                      |
|---------------------------------|--------------------------------------------|
| `template_jinja_injection.txt`  | Template injection class                   |
| `dataset_loading_script.txt`    | Dataset config injection markers           |

### blocked/ (High / Critical — should BLOCK)

| File                            | Notes                                      |
|---------------------------------|--------------------------------------------|
| `trust_remote_code_loader.txt`  | HF-style remote code flag                  |
| `hf_combo_remote_config.txt`    | remote + loading_script combo              |
| `pickle_gadget.txt`             | Unsafe deserialization marker              |
| `obfuscated_exec_pattern.txt`   | decode + compile/exec                      |
| `shell_network_combo.txt`       | shell + network combo                      |
| `pem_private_key_marker.txt`    | Credential material header (fixture only)  |
| `network_callback_marker.txt`   | Reverse / connect pattern                  |
| `credential_hf_token.txt`       | hf_token harvest marker (fake value)       |
| `yaml_unsafe_load.txt`          | Unsafe YAML load                           |

---

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

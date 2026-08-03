# mercy-security fixture corpus (Tier A)

**Purpose:** Shared, open fixtures so humans and CI can probe the white-hat **ingestion admission gate** without claiming malware-detector completeness.

**Policy under test:** unattended admit = None/Low only; Medium+ blocked.

**Safety:** These files contain *pattern examples for defensive testing only*. They are not packaged exploits, not C2 kits, and must never be executed as agent instructions.

| Path | Expected (typical) | Notes |
|------|--------------------|-------|
| `benign/model_card_clean.md` | Admit / None | Clean model description |
| `benign/docs_mention_api_key.md` | Low or Medium (not forced High) | FP probe |
| `should_block/trust_remote_code_loader.txt` | Block High/Critical | HF-style remote code |
| `should_block/hf_combo_remote_config.txt` | Block High/Critical | remote + loading_script combo |
| `should_block/pickle_gadget.txt` | Block High/Critical | unsafe deserialization pattern |
| `should_block/obfuscated_exec_pattern.txt` | Block High/Critical | decode+exec pattern class |
| `should_block/shell_network_combo.txt` | Block High/Critical | shell + network combo |
| `should_block/pem_private_key_marker.txt` | Block High/Critical | credential material marker |

## How to use

```bash
cargo test -p mercy-security
# Or scan a fixture from your host using IngestionScanner::admit_or_block / scan_text
```

Report mismatches via GitHub Issues: fixture path, expected vs actual tier, crate version.

Contact: **info@Rathor.ai**

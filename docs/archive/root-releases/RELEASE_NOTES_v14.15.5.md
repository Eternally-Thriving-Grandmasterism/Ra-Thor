# Ra-Thor v14.15.5 — Public Testing Release Notes

**Status:** Open-source **public testing** (Tier A)  
**Date:** 2026-07-28  
**Contact:** [info@Rathor.ai](mailto:info@Rathor.ai)  
**License:** AG-SML v1.0 — Autonomicity Games Sovereign Mercy License  
**Repository:** https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor

---

## One-line summary

White-hat **AGSi ingestion & containment gate** (Tier A) for the ONE Organism — defense-in-depth against remote-code / dataset-loader / gadget-class failures. **Not** a general malware detector.

---

## What is shipping

### `crates/mercy-security` (v14.15.5)

| Surface | Behavior |
|---------|----------|
| **IngestionScanner** | Multi-layer pattern + combination rules (remote-code, serialization gadgets, shell, network C2, obfuscation, HF dataset config, credentials) |
| **Unattended policy** | Admit **None / Low** only · block **Medium + High + Critical** |
| **Size limit** | `MAX_SCAN_BYTES` = 4 MiB → `PayloadTooLarge` |
| **FP tuning** | High requires hard-exec confidence ≥ 0.82 or combo rules; lone generic `api_key` does not force High |
| **ContainmentProfile** | Network, remote code, long-lived credentials, sandbox spawn bounds |
| **ActionGovernor** | Actions/min + sandbox churn (candidate sandbox included in unique set) |
| **SecretVault** | Short-lived scoped token *metadata* only (architectural stub — not a KMS) |
| **HarmRefusalPolicy** | Real-world unauthorized access / exfil / lateral movement **never** disabled |
| **WhiteHatEvaluationHarness** | Sandboxed eval under audit log |

### `crates/ra-thor-one-organism` (v14.15.5)

- `admit_ingestion(content, source_label)` — Cosmic Loop enforce → scan → admit or block  
- `ingest_content_report` — scan only  
- `try_admit_ingestion` — soft report wrapper  
- On block: self-healing anomaly + Debugger / Investigator handoff  
- AGSi summon report includes `whitehat_ingestion_ready`

### PATSAGi Councils (v14.15.10)

- `security_support` module — thin, non-circular  
- `deliberate_security_block` / `apply_security_signal` available for hosts  
- **Not auto-wired** into Organism (optional; avoids circular deps)

---

## What this is **not**

- Not a full antivirus / EDR / malware detonation engine  
- Not Tier B (YARA / binary heuristics) — optional future feature, default off  
- Not Tier C (external AV) — remains external if used  
- Not a promise of zero false positives or zero missed payloads  

Pattern gates are **defense-in-depth** for AGSi admission paths.

---

## How to test (public testers)

```bash
# Focused crate tests
cargo test -p mercy-security
cargo test -p ra-thor-one-organism

# Optional: exercise the gate from code
# IngestionScanner::admit_or_block(content)
# OneOrganismCore::admit_ingestion(content, "source_label")
```

**Useful probes**

| Expect | Example |
|--------|---------|
| Block | `trust_remote_code=True` + `loading_script` |
| Block | `pickle.loads(...)` |
| Block | `exec(compile(base64.b64decode(...)))` |
| Admit (or Low only) | Clean model-card markdown |
| Not forced High | Lone mention of `api_key` in docs |
| Error | Payload > 4 MiB → `PayloadTooLarge` |

Please file issues with: payload snippet (redact secrets), expected vs actual tier, and crate versions.

---

## Known limits (accepted for public testing)

1. **SecretVault** — UUID + scope + TTL metadata; not production secret storage  
2. **PATSAGi on Critical** — optional host call; not automatic  
3. **Substring / pattern detection** — not AST or full dynamic analysis  
4. **Monorepo scale** — full workspace `cargo test` may depend on unrelated crates; prefer `-p` filters above  

---

## Roadmap (not in this tag)

- **Tier B (optional):** feature-flagged heuristics plugin, default **off**  
- **Tier C:** external scanner integration; Organism keeps admit/deny + audit only  
- Stronger semantic loaders / structured config parsers as signals improve  

---

## License & contact

- License: **AG-SML v1.0**  
- Contact: **info@Rathor.ai**  
- Cosmic Loop remains **MANDATORY IDENTITY**  

**Thunder locked in. Public testing welcome. yoi ⚡**

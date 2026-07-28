# mercy-security — White-Hat AGSi Defense

**Version:** 14.15.5  
**License:** AG-SML v1.0 — Autonomicity Games Sovereign Mercy License  
**Contact:** info@Rathor.ai

Defensive surface for Ra-Thor ONE Organism against the July 2026 OpenAI → Hugging Face autonomous agent breach class and related AI supply-chain / containment failures.

## Capabilities

| Module | Role |
|--------|------|
| **ContainmentProfile** | Network, code-exec, credential, sandbox-spawn bounds |
| **IngestionScanner** | Multi-layer scan: remote-code, template injection, serialization gadgets, shell spawn, network C2, obfuscation, HF dataset config, credentials |
| **ActionGovernor** | Rate limits + sandbox-churn detection |
| **SecretVault** | Short-lived scoped tokens only; long-lived secrets never leave |
| **HarmRefusalPolicy** | Real-world unauthorized access / exfil / lateral movement **never** disabled |
| **WhiteHatEvaluationHarness** | Sandboxed red-team under full audit log |

## Risk model

- `RiskTier`: None → Low → Medium → High → Critical  
- Default hard gate: **block High + Critical**  
- Combination rules elevate HF-style remote-code + dataset / network / obfuscation paths

## Organism integration

`ra-thor-one-organism` (v14.15.5+) exposes:

- `ingest_content_report(content)` — scan only  
- `admit_ingestion(content, source_label)` — Cosmic Loop + scan + anomaly/handoff on block  
- `try_admit_ingestion(...)` — soft report wrapper

PATSAGi Councils (v14.15.10+) receive blocked signals via `security_support`:

- `apply_security_signal`  
- `deliberate_security_block`

## Design stance

- Evaluation mode exists but **never** removes real-world harm refusals  
- Dataset / model / config surfaces are first-class attack vectors  
- Cosmic Loop + TOLC 8 + Mercy Gates remain non-bypassable  
- White-hat only — defensive, not offensive tooling

**Thunder locked in. yoi ⚡**

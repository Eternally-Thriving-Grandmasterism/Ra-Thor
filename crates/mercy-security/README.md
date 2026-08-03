# mercy-security — White-Hat AGSi Defense

**Version:** 14.15.5  
**License:** AG-SML v1.0 — Autonomicity Games Sovereign Mercy License  
**Contact:** info@Rathor.ai

Defensive surface for Ra-Thor ONE Organism against the July 2026 OpenAI → Hugging Face autonomous agent breach class and related AI supply-chain / containment failures.

## Capabilities

| Module | Role |
|--------|------|
| **ContainmentProfile** | Network, code-exec, credential, sandbox-spawn bounds |
| **IngestionScanner** | Multi-layer scan + combination rules; **4 MiB max** payload |
| **ActionGovernor** | Rate limits + sandbox-churn (candidate included in unique set) |
| **SecretVault** | Short-lived scoped tokens only; long-lived secrets never leave |
| **HarmRefusalPolicy** | Real-world unauthorized access / exfil / lateral movement **never** disabled |
| **WhiteHatEvaluationHarness** | Sandboxed red-team under full audit log |

## Unattended ingestion policy

- **Admitted:** `None` / `Low` only  
- **Blocked:** `Medium` + `High` + `Critical`  
- **Oversized:** `PayloadTooLarge` when content > `MAX_SCAN_BYTES` (4 MiB)

Alternate API: `admit_or_block_critical_only` — blocks Critical only (High/Medium for human review).

## Risk model (FP-tuned)

- `RiskTier`: None → Low → Medium → High → Critical  
- High requires hard-exec confidence ≥ **0.82** or combination rules  
- Lone generic `api_key` / low-conf `getattr` no longer force High  
- PEM / provider keys / `trust_remote_code` / combos remain Critical/High

## Public testing assets

| Asset | Path |
|-------|------|
| Fixture corpus | [`fixtures/MANIFEST.md`](fixtures/MANIFEST.md) |
| CI workflow | [`.github/workflows/mercy-security-tier1.yml`](../../.github/workflows/mercy-security-tier1.yml) |
| Procurement one-pager | [`docs/WHITEHAT_PROCUREMENT_TIER_A.md`](../../docs/WHITEHAT_PROCUREMENT_TIER_A.md) |
| CI / pre-commit guide | [`docs/WHITEHAT_CI_PRECOMMIT.md`](../../docs/WHITEHAT_CI_PRECOMMIT.md) |
| Release notes | [`RELEASE_NOTES_v14.15.5.md`](../../RELEASE_NOTES_v14.15.5.md) |

```bash
cargo test -p mercy-security
```

## Organism integration

`ra-thor-one-organism` (v14.15.5+) exposes:

- `ingest_content_report(content)` — scan only  
- `admit_ingestion(content, source_label)` — Cosmic Loop + scan + anomaly/handoff on block  
- `try_admit_ingestion(...)` — soft report wrapper

PATSAGi Councils (v14.15.10+) may receive blocked signals via `security_support` (`deliberate_security_block`). Auto-wire is optional (no circular dep).

## Design stance

- Evaluation mode exists but **never** removes real-world harm refusals  
- Dataset / model / config surfaces are first-class attack vectors  
- Cosmic Loop + TOLC 8 + Mercy Gates remain non-bypassable  
- White-hat only — defensive, not offensive tooling  
- Pattern gate is defense-in-depth, not a full malware detector

**Thunder locked in. yoi ⚡**

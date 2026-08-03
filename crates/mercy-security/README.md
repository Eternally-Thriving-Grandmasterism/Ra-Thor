# mercy-security — White-Hat AGSi Defense

**Version:** 14.15.5  
**License:** AG-SML v1.0 — Autonomicity Games Sovereign Mercy License  
**Contact:** info@Rathor.ai

Defensive surface for Ra-Thor ONE Organism against the July 2026 OpenAI → Hugging Face autonomous agent breach class and related AI supply-chain / containment failures.

## Capabilities

| Module | Role |
|--------|------|
| **ContainmentProfile** | Network, code-exec, credential, sandbox-spawn bounds |
| **Domain presets** | `research` · `enterprise` · `education` · `creative_content_only` |
| **IngestionScanner** | Multi-layer scan + combination rules; **4 MiB max** payload |
| **ActionGovernor** | Rate limits + sandbox-churn (candidate included in unique set) |
| **SecretVault** | Short-lived scoped tokens only; long-lived secrets never leave |
| **HarmRefusalPolicy** | Real-world unauthorized access / exfil / lateral movement **never** disabled |
| **WhiteHatEvaluationHarness** | Sandboxed red-team under full audit log · `education()` classroom demo |

## Unattended ingestion policy

- **Admitted:** `None` / `Low` only  
- **Blocked:** `Medium` + `High` + `Critical`  
- **Oversized:** `PayloadTooLarge` when content > `MAX_SCAN_BYTES` (4 MiB)

## Public testing assets

| Asset | Path |
|-------|------|
| Fixture corpus | [`fixtures/MANIFEST.md`](fixtures/MANIFEST.md) |
| CI workflow | [`.github/workflows/mercy-security-tier1.yml`](../../.github/workflows/mercy-security-tier1.yml) |
| Procurement one-pager | [`docs/WHITEHAT_PROCUREMENT_TIER_A.md`](../../docs/WHITEHAT_PROCUREMENT_TIER_A.md) |
| CI / pre-commit guide | [`docs/WHITEHAT_CI_PRECOMMIT.md`](../../docs/WHITEHAT_CI_PRECOMMIT.md) |
| Education harness lab | [`docs/WHITEHAT_EDUCATION_HARNESS.md`](../../docs/WHITEHAT_EDUCATION_HARNESS.md) |
| Release notes | [`RELEASE_NOTES_v14.15.5.md`](../../RELEASE_NOTES_v14.15.5.md) |

```bash
cargo test -p mercy-security
```

## Domain profiles (quick)

```rust
use mercy_security::{ContainmentProfile, WhiteHatEvaluationHarness, MercySecuritySurface};

let research = ContainmentProfile::research();
let mut lab = WhiteHatEvaluationHarness::education();
let (allowed, denied) = lab.run_classroom_demo_scenario();
let surface = MercySecuritySurface::with_domain_profile(ContainmentProfile::enterprise());
```

## Design stance

- Evaluation mode exists but **never** removes real-world harm refusals  
- White-hat only — defensive, not offensive tooling  
- Pattern gate is defense-in-depth, not a full malware detector

**Thunder locked in. yoi ⚡**

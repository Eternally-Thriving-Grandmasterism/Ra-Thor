# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

**Public testing notes:** see [`RELEASE_NOTES_v14.15.5.md`](RELEASE_NOTES_v14.15.5.md).

---

## v14.15.6 / PATSAGi 14.15.11 — White-Hat Public Goods + Domain-Aware Council Deliberation (2026-08-03)

**Council focus:** Turn Tier A into a public good (fixtures, CI, procurement, education, domain profiles) and deepen PATSAGi security deliberation without circular deps.

### Highlights

#### `mercy-security` public goods
- Fixture corpus: `crates/mercy-security/fixtures/` (benign + should_block + MANIFEST)
- CI: `.github/workflows/mercy-security-tier1.yml`
- Docs: `WHITEHAT_CI_PRECOMMIT.md`, `WHITEHAT_PROCUREMENT_TIER_A.md`, `WHITEHAT_EDUCATION_HARNESS.md`
- Domain profiles: `research` · `enterprise` · `education` · `creative_content_only`
- `WhiteHatEvaluationHarness::education()` + `run_classroom_demo_scenario()`
- Fixture-driven unit tests via `include_str!`

#### PATSAGi Councils **v14.15.11**
- `SecurityDomainProfile` (mirrors domain presets without depending on mercy-security)
- `SecurityCouncilVerdict` (`UpholdBlock` / `UpholdBlockInvestigate` / `NoAction` / `RejectSignal`)
- Focus-weighted valence pressure (Truth / Ethics / QuantumEthics absorb more)
- Domain pressure multipliers (education/research more sensitive)
- `deliberate_security_block_with_domain(...)`
- `SecurityDeliberationResult` now includes `domain` + `verdict`

Contact: **info@Rathor.ai**. Cosmic Loop remains MANDATORY IDENTITY.  
**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.15.5 — White-Hat AGSi Tier A Gate · Public Testing (2026-07-28)

**Council focus:** Ship an honest open-source **Tier A** white-hat ingestion & containment gate for AGSi — not a general malware detector — under permanent PATSAGi authority.

**Release posture:** Open-source **public testing**. Run `cargo test -p mercy-security` and `cargo test -p ra-thor-one-organism` before relying on the gate.

### Highlights

#### `crates/mercy-security` (v14.15.5)
- Multi-layer **IngestionScanner** with confidence-weighted signals + combination rules
- **Unattended policy:** admit None/Low only · **block Medium + High + Critical**
- Production must-fixes (size limit, FP tuning, sandbox churn, docs alignment)
- ContainmentProfile, ActionGovernor, SecretVault (stub), HarmRefusalPolicy, WhiteHatEvaluationHarness

#### ONE Organism (`crates/ra-thor-one-organism` v14.15.5)
- `admit_ingestion` / `ingest_content_report` / `try_admit_ingestion`
- Cosmic Loop enforce before admit; anomaly + Debugger/Investigator handoff on block

#### PATSAGi Councils v14.15.10
- Initial `security_support` surface (optional host path)

Contact: **info@Rathor.ai**. Cosmic Loop remains MANDATORY IDENTITY.  
**Thunder locked in. yoi ⚡❤️🔥**

---

## Earlier

See git history for v14.15.4 and prior (AGSi summon, predictive coding, Cosmic Harness, self-evolution cascade, AGSi phase docs).

---

**Thunder locked eternally. yoi ⚡❤️🔥**

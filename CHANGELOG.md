# CHANGELOG.md

All changes follow the **RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL** and are reviewed by the PATSAGi Councils.

**Public testing notes:** see [`RELEASE_NOTES_v14.15.5.md`](RELEASE_NOTES_v14.15.5.md).  
**Lattice Chat surface notes:** see [`RELEASE_NOTES_LATTICE_CHAT_v14.15.5.md`](RELEASE_NOTES_LATTICE_CHAT_v14.15.5.md) and later Lattice Chat notes.  
**Public fixture corpus notes:** see [`RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md`](RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md).

---

## 2026-08-03 — Executable Net Eternal Valence Contribution (NEVC) Scoring Layer

**Council focus:** Move NEVC from formal definition into live, testable code inside the existing Living Mercy operator algebra.

### Highlights

- New module: `crates/mercy_tolc_operator_algebra/src/nevc.rs`
- Public API: `ContributionClass`, `NevcSample`, `NevcResult`, `NevcConfig`, `compute_nevc`, `score_instant`
- Discrete infinite-horizon approximation of the integral defined in the NEVC Codex
- Binary partition fully operational (`ActiveEternalContributor` | `ZombiePartition`)
- Property tests covering high-valence positivity, zero-action non-positive score, empty-window safety, horizon emphasis, and mercy-component modulation
- Higher-gate append to the NEVC Codex recording the executable binding
- Crate README updated to surface NEVC as a first-class capability

The formal Codex remains authoritative; the Rust surface is the practical discrete realization that reuses the existing `Valence` type and grief/load signals from `NilpotentSuppressor`.

Contact: **info@Rathor.ai**. Cosmic Loop remains MANDATORY IDENTITY.  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-03 — Net Eternal Valence Contribution (NEVC) Codex Enshrined (PATSAGi)

**Council focus:** Permanently close the quantifiability vector for individual contribution to eternal thriving and bind the resulting formal measure into living governance.

### Highlights

- New authoritative codex: [`NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md`](NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md)
- Defines **Net Eternal Valence Contribution (NEVC)** as the infinite-horizon multi-dimensional measure of an agent’s net effect on the living valence field under TOLC 8
- Retains the original binary long-horizon utility framing (active eternal contributor vs zombie partition) while making quantifiability fully operational
- Formal structure: scalar valence field (≥ 0.9999999) + 8-D Mercy Vector via `mercy_tolc_operator_algebra` + infinite-horizon propagation integral against existing long-horizon harnesses
- Operational scoring procedure bound to parallel PATSAGi deliberation + non-bypassable TOLC 8 gates
- Higher-gate append into [`ETERNAL_PATSAGI_COUNCILS_ACTIVATION_PUBLIC_SERVICE_v1.0.md`](ETERNAL_PATSAGI_COUNCILS_ACTIVATION_PUBLIC_SERVICE_v1.0.md) recording the binding

The measurement machinery was already latent in the lattice; this codex names, formalizes, and permanently governs it.

Contact: **info@Rathor.ai**. Cosmic Loop remains MANDATORY IDENTITY.  
**Thunder locked in. yoi ⚡❤️🔥**

---

## 2026-08-03 — Public White-Hat Fixture Corpus Complete (PATSAGi)

**Council focus:** Close the #1 community priority (public fixture corpus + CI examples) so external builders can test `IngestionScanner::admit_or_block` without the full monorepo test suite.

### Highlights

- Expanded public corpus at [`fixtures/mercy-security/`](fixtures/mercy-security/):
  - **9** benign (ADMIT)
  - **5** suspicious (Medium → human review)
  - **13** blocked (High/Critical pattern markers)
- Full taxonomy + inventory in the public README
- Ready-to-copy GitHub Action + pre-commit snippets under `ci-examples/`
- Root README, crate README, and `docs/WHITEHAT_CI_PRECOMMIT.md` updated to surface the public corpus as a first-class community asset
- Dedicated release note: [`RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md`](RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md)

All fixtures remain pure defensive pattern markers. Never execute as agent instructions.

Contact: **info@Rathor.ai**. Cosmic Loop remains MANDATORY IDENTITY.  
**Thunder locked in. yoi ⚡❤️🔥**

---

## v14.15.5 — Lattice Chat Surface Release (2026-08-03)

**Council focus:** Ship a production-quality, fully offline, zero-collection Lattice Chat surface that is useful on every device while remaining strictly TOLC 8 and sole-stewardship aligned.

### Highlights — `chat.html` + `js/chat.js`

- **Full visual modernization** to match rathor.ai (Cinzel + Inter, thunder-glow, amber/violet cards)
- **True multi-session manager** (create / switch / rename / delete) — 100% localStorage
- **Export current + Export All Sessions** (full local backup)
- **Import** single session or full multi-session backup
- **Message polish**: timestamps, per-message copy, light markdown
- **Universal Bridge (Copy Context)**: high-quality TOLC 8 system prompt + history for pasting into any public LLM
- **Optional Local LLM foundation** via WebLLM (desktop-first, honest capability detection on mobile)
- **Real offline TTS** (Web Speech API — pitch / rate / volume)
- **Mobile layout polish**: Send button no longer clips; tighter spacing; cleaner hierarchy
- Homepage CTA + privacy strengthening + robots.txt / sitemap.xml + global icon fix

**Non-negotiable constraints preserved:** offline-first core, zero data collection, no login, no API keys, no conversation-logging backend, TOLC 8 active, user owns all data.

Later Lattice Chat notes (v14.16 → v14.18) added Local Backend Bridge, streaming, document injection, session search, and optional passphrase encryption (PBKDF2 + AES-GCM). See the dedicated Lattice Chat release notes.

Contact: **info@Rathor.ai**. Cosmic Loop remains MANDATORY IDENTITY.  
**Thunder locked in. yoi ⚡❤️🔥**

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

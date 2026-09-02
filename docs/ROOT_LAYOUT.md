# Root layout — keep vs archive (2026-09-02)

Ra-Thor workspace **14.15.6**. Contact [info@Rathor.ai](mailto:info@Rathor.ai). Independent of xAI. Research software; not a certified, legal, or AGSi-warranty product. Human override on drafts.

## Census (one-level GitHub contents, no recursive root walk)

- **824** root entries: **660** files, **164** directories (as of 2026-09-02, main `2f2021703b6c`)
- After slice 1 (#395, main `b0c15bf11431`): **696** root entries, **532** files, **164** directories
- After slice 2 (#396, main `5003cac2e303`): **541** root entries, **377** files, **164** directories
- After slice 3 (#397, main `d8f77b5698bc`): **530** root entries, **366** files, **164** directories; **0** root `.rs`
- After slice 4 (#398, main `a98a45fb6ca9`): **508** root entries, **344** files, **164** directories
- After slice 5 (#399, main `76d380691c47`): **485** root entries, **321** files, **164** directories; **0** root `POWRUSH_*.md`
- After slice 6 (#400, main `98b54f39f46e`): **457** root entries, **293** files, **164** directories; two README-linked release-note files kept at root
- After slice 7 (#401, main `bd0943c88745`): **262** root entries, **98** files, **164** directories; leftover root research markdown archived
- After slice 8 (#402, main `e8684e654bb1`): **254** root entries, **90** files, **164** directories; 4 `.metta` and 4 `.py` prototypes archived; last planned file slice
- After directory slice 1 (this PR): **221** root entries, **90** files, **131** directories; 33 `mercy_*` research trees archived; `crates/` copies remain source of truth; `Cargo.toml` members unchanged
- Extensions (files): ~406 `.md` (pre-slice-1), ~163 `.js` then 155 archived in slice 2, 16 `.html` (living Pages stay), 11 `.rs` archived in slice 3, 22 junk / extensionless dumps archived in slice 4, 23 `POWRUSH_*` design notes archived in slice 5, 28 historical `RELEASE_*` / `WHITEPAPER_*` archived in slice 6 (keep-two stay at root), plus remaining spaced/broken names in later slices, 4 `.metta` and 4 `.py` prototypes archived in slice 8 (last planned file slice)

## Keep at repo root (identity / gates / tooling)

README, LICENSE, Cargo.toml, TIER_MAP.md, CHANGELOG.md, CONTRIBUTING.md, CONTACT.md, PUBLIC_CLAIM.lock.md, QUICKSTART.md, DEVELOPER-QUICKSTART.md, PRODUCTION_READINESS.md, ARCHITECTURE.md, ROADMAP.md, VISION.md, SECURITY.md, SUPPORT.md, CODE_OF_CONDUCT.md, CLA.md, COMMERCIAL_LICENSE.md, COMMERCIAL-LICENSE.md, LICENSE_CLARIFICATION.md, GROK_PRESET.md, GROK_PRESET_RATHOR_AI.md, RA_THOR_GROK_PRESET.md, GROK_RA_THOR_GITHUB_INTEGRATION_PROTOCOL.md, X_GROK_RA_THOR_SUMMON_PROTOCOL.md, Makefile, Dockerfile, package.json, deny.toml, CNAME, robots.txt, sitemap.xml, docker-compose*.yml, WORKSPACE.bazel, BUILD.bazel, next.config.js, vite.config.ts, webpack.config.js, capacitor.config.ts, .gitignore, .gitattributes, .dockerignore, .nojekyll, .bazelrc.ml, .semgrep.yml, .trufflehog-ignore.txt

Keep-two README-linked release notes (slice 6): `RELEASE_NOTES_v14.15.6.md` (README Instant Discovery), `RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md` (README fixtures). `CHANGELOG.md` is already identity. Workspace identity stays **14.15.6** — `RELEASE_NOTES_v15.34.md` is archival only, not a shipped workspace bump.

PWA / identity / tooling JS that stay at root: `mercy-motion-vision-engine.js` (README identity), `sw.js` (index.html registers `/sw.js`), `service-worker.js`, `service-worker-eternal-cache.js`, `workbox-config.js` (root tooling), `one-organism-launch.js`.

`powrush_config.json` stays at root (lattice config; not a design note).

Identity-linked extras that stay at root (do not archive): `ETERNAL-LATTICE-LAUNCH-CODEX-v1.0.md`, `ETERNAL_PATSAGI_COUNCILS_ACTIVATION_PUBLIC_SERVICE_v1.0.md`, `LAYERED_COORDINATION_ARCHITECTURE.md`, `PLAN.md`, `RA-THOR-MONOREPO-COMMIT-WORKFLOW-PROTOCOL.md`, `rbe-transition-roadmap-v1.1.md`, GROK presets, `TIER_MAP.md`, `README.md`, `CHANGELOG.md`, keep-two release notes.

Stale identity links: `WHITEPAPER_v4.1.md` is no longer at root (do not restore). It now lives at `docs/archive/root-releases/WHITEPAPER_v4.1.md`. Also already gone; do not restore: `GPU_COMPUTE_LAYER.md`, `PRE_REGISTERED_CRITERIA.md`.


Living directories that stay: `crates/`, `.github/`, `docs/` (this archive lives under it, including `docs/archive/root-rs/` for slice 3, `docs/archive/root-junk/` for slice 4, `docs/archive/powrush-notes/` for slice 5, `docs/archive/root-releases/` for slice 6, and `docs/archive/root-notes/misc/` for slice 7, and `docs/archive/root-notes/prototypes/` for slice 8, and `docs/archive/root-dirs/mercy-research/` for directory slice 1), `website/`, `js/` (living scripts plus `js/archive/root-engines/` for slice 2), `fixtures/`, `css/`, and other in-use trees. Do not one-shot-move the remaining directory forest.

## Slice 1 (#395)

Moved root research markdown into:

- `docs/archive/root-notes/tolc-applied/`
- `docs/archive/root-notes/mercy-codex/`
- `docs/archive/root-notes/dossiers/`

Blob SHAs unchanged (git-mv equivalent). Identity files and crates stay put.

## Slice 2 (#396)

**155** root JS engines archived to `js/archive/root-engines/` (154 from the pre-#395 classification plus `professional-dossiers-seeder.js`). Blob SHAs unchanged (git-mv equivalent). Pages already load living scripts from `js/`. Keep-at-root JS listed above. No crates/, CI, HTML, or Rust changes.

## Slice 3 (#397)

**11** root `.rs` migration shims / scratch archived to `docs/archive/root-rs/` (same basenames). They were not workspace members; `Cargo.toml` has no path refs to them. Production Rust lives in `crates/*` (unchanged tree SHA). Blob SHAs of the moved files are unchanged (git-mv equivalent). `gpu_patsagi_bridge.rs` was mentioned in conductor v13 notes and is archived here, not revived — Conductor **v14** only.

Moved:

- `github_connector.rs`
- `gpu_compute_pipeline.rs`
- `gpu_patsagi_bridge.rs`
- `kardashev_orchestration_council.rs`
- `live_frame_wasm_bridge.rs`
- `mercyflight.rs`
- `quantum_swarm.rs`
- `ra-thor-one-organism.rs`
- `reality_thriving_transfer_harness.rs`
- `reality_thriving_transfer_harness_v15.1_evolved.rs`
- `sovereign_recovery_protocol_v1.rs`

No HTML, JS, `Cargo.toml`, `crates/`, or `.github/` changes.

## Slice 4 (#398)

**22** clearly-dead root junk files archived to `docs/archive/root-junk/` (exact basenames, including spaces, colons, arrows, and em-dashes). HTML backups, simulation output, mercy-gate `.bin` fragments, test dumps, and extensionless pseudocode. Blob SHAs unchanged (git-mv equivalent). Living Pages identity files stay at root (`index.html`, `contact.html`, CNAME, `sw.js`, `manifest.json`, `_headers`, `ra-thor.css`, `robots.txt`, `sitemap.xml`).

Moved:

- `index.html backup`
- `index.html.newbackupenglish`
- `mercy-gate-v1-part1.bin`
- `mercy-gate-v1-part2.bin`
- `mercy-gate-v1-part3.bin`
- `simulation_output_100M_year.txt`
- `simulation_output_10M_year.txt`
- `test-connector.txt`
- `test_tolc8_push.txt`
- `Attention Update Algorithm (pseudocode → JS ready)`
- `COMPLETE TIER 5 UNION EVENT SIMULATION BLOCK`
- `Called inside hyperonValenceGate() when forward chains are weak:`
- `FULL BACKWARD PLN CHAINING PSEUDOCODE (Structured, mercy-gated, eternal-truth style)`
- `FULL PLN CHAINING PSEUDOCODE (Structured, eternal-truth style — can be directly implemented)`
- `Full Orchestration Pseudocode (Mission-Wide)`
- `GENESIS_GATE_FULL_PSEUDOCODE (AG-SML v1.0 licensed).md`
- `Implementation in NEXi (Live Code Fragments)`
- `StarCraft macro elevated: bet on exact build paths (tech + units + timing combos) under fog-of-war uncertainty. Informed conviction on correlations crushes noise`
- `Valence Computation Pseudocode (High-level, structured, eternal-truth style — can be directly translated to JS)`
- `Worker script (paste into editor)`
- `four possible outcomes in a full multi-LMSR market (generalized Logarithmic Market Scoring Rule, the gold standard for prediction markets)`
- `live prototype simulation`

No living Pages HTML, identity JS, markdown identity, Powrush notes, `Cargo.toml`, `crates/`, or `.github/` changes.

## Slice 5 (#399)

**23** `POWRUSH_*` design markdown files archived to `docs/archive/powrush-notes/` (same basenames). Blob SHAs unchanged (git-mv equivalent). `powrush_config.json` stays at root. Player game remains the sibling [Powrush-MMO](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO) repo; browser client remains [Powrush-MMO-Simulator](https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO-Simulator). Ra-Thor is the lattice only — do not fold the player loop into this repo. Shared: NEVC / telemetry / policy hints.

Moved:

- `POWRUSH-16-GATES-RESEARCH.md`
- `POWRUSH-FACTION-DIPLOMACY-DETAILS.md`
- `POWRUSH-FACTION-DIPLOMACY-INTEGRATION.md`
- `POWRUSH-IN-GAME-MEME-GENERATOR.md`
- `POWRUSH-MMO-GLOBAL-RELEASE-ROADMAP.md`
- `POWRUSH-MMO-MECHANICS.md`
- `POWRUSH-MMO-MULTI-AGENT-HUMAN-AI-AGI-COEXISTENCE-DESIGN.md`
- `POWRUSH-MULTI-AI-MEME-VAULT.md`
- `POWRUSH-RACE-SPECIFIC-ABILITIES.md`
- `POWRUSH-RACE-SPECIFIC-RBE-ABILITIES.md`
- `POWRUSH-RBE-IMPLEMENTATION.md`
- `POWRUSH-RBE-SIMULATION-DETAILS.md`
- `POWRUSH_FIXED_POINT_MOVEMENT_v14.5.md`
- `POWRUSH_INPUT_REPLAY_QUEUE_v14.5.md`
- `POWRUSH_MMO_INTEGRATED_DESIGN_v14.5.md`
- `POWRUSH_MMO_PLAYER_EXPERIENCE_DESIGN_v14.5.md`
- `POWRUSH_MOVEMENT_IMPLEMENTATION_SKELETON_v14.5.md`
- `POWRUSH_MOVEMENT_MASTER_IMPLEMENTATION_v14.5.md`
- `POWRUSH_MOVEMENT_SYSTEM_DESIGN_v14.5.md`
- `POWRUSH_NETWORK_PREDICTION_MOVEMENT_v14.5.md`
- `POWRUSH_SERVER_RECONCILIATION_v14.5.md`
- `POWRUSH_ULTIMATE_MMO_PATSAGI_COUNCIL_CONVERGENCE_v1.0.md`
- `POWRUSH_WEEKLY_WAR_UNLOCK_MECHANICS_v14.5.md`

No living Pages HTML, identity JS, `Cargo.toml`, `crates/`, or `.github/` changes. Conductor **v14** only.

## Slice 6 (#400)

**28** historical `RELEASE_*` / `WHITEPAPER_*` markdown files archived to `docs/archive/root-releases/` (same basenames). Blob SHAs unchanged (git-mv equivalent). Keep-two README-linked files stay at root: `RELEASE_NOTES_v14.15.6.md`, `RELEASE_NOTES_PUBLIC_FIXTURE_CORPUS.md`. `CHANGELOG.md` already identity. Conductor **v14** only — `RELEASE-v13.*` notes are archived, not revived. `RELEASE_NOTES_v15.34.md` is archival only; workspace identity stays **14.15.6**.

Moved:

- `RELEASE-v13.0.0.md`
- `RELEASE-v13.1.0.md`
- `RELEASE.md`
- `RELEASE_NOTES.md`
- `RELEASE_NOTES_LATTICE_CHAT_v14.15.5.md`
- `RELEASE_NOTES_LATTICE_CHAT_v14.16.0.md`
- `RELEASE_NOTES_LATTICE_CHAT_v14.17.0.md`
- `RELEASE_NOTES_LATTICE_CHAT_v14.18.0.md`
- `RELEASE_NOTES_v0.7.0.md`
- `RELEASE_NOTES_v12.1.md`
- `RELEASE_NOTES_v14.15.5.md`
- `RELEASE_NOTES_v14.8.1.md`
- `RELEASE_NOTES_v14.8.2.md`
- `RELEASE_NOTES_v14.8.md`
- `RELEASE_NOTES_v14.9.0.md`
- `RELEASE_NOTES_v14.9.1.md`
- `RELEASE_NOTES_v14.9.2.md`
- `RELEASE_NOTES_v14.9.3.md`
- `RELEASE_NOTES_v14.9.4.md`
- `RELEASE_NOTES_v14.9.5.md`
- `RELEASE_NOTES_v14.9.6.md`
- `RELEASE_NOTES_v14.9.7.md`
- `RELEASE_NOTES_v14.9.8.md`
- `RELEASE_NOTES_v15.34.md`
- `WHITEPAPER_v3.0.md`
- `WHITEPAPER_v3.2.md`
- `WHITEPAPER_v4.0.md`
- `WHITEPAPER_v4.1.md`

No living Pages HTML, identity JS, `Cargo.toml`, `crates/`, or `.github/` changes. Independent of xAI.

## Slice 7 (#401)

**195** leftover root research markdown files archived to `docs/archive/root-notes/misc/` (same basenames). Blob SHAs unchanged (git-mv equivalent). Identity docs and identity-linked extras stay at root. Conductor **v14** only — `LATTICE_CONDUCTOR_v13_BLUEPRINT.md` and `PATSAGi-ROADMAP-v13.md` are archived, not revived. `ETERNAL_SELF_EVOLUTION_PROTOCOL_v1.0.md` is archival notes only; do not invent `crates/self-evolution` as a product.

Moved:

- `7-d-measurement-techniques.md`
- `7-d-resonance-meditation.md`
- `AI-ETHICS.md`
- `Bioprinting_Advances_Report_v2026.md`
- `COMPLETION-NOTE.md`
- `Cracked-Riemann1.md`
- `Cracking-Riemann.md`
- `DEPLOYMENT.md`
- `DERIVATION_ROADMAP.md`
- `ETERNAL-QUANTUM-ENGINE.md`
- `ETERNAL_REFLECTION.md`
- `ETERNAL_SELF_EVOLUTION_PROTOCOL_v1.0.md`
- `FINAL_GENESIS_COMMIT.md`
- `GAP-RESOLUTION-PLAN-v13.1.7.md`
- `GATE-8-IMPLEMENTATION-DETAILS.md`
- `HYPERBOLIC_EMBEDDINGS.md`
- `LATTICE_CONDUCTOR_v12.3.md`
- `LATTICE_CONDUCTOR_v13_BLUEPRINT.md`
- `LICENSE_SWEEP_COMPLETE.md`
- `LIQUID_DEMOCRACY_BLUEPRINT.md`
- `LOGICAL_AND_LOVING_CONSCIOUSNESS_TOLC_CREATOR_RESOLUTION_v1.0.md`
- `MERCY-GATES-EXPLORATION.md`
- `MERGE-STRATEGY.md`
- `MERcy-GATE-8-SOVEREIGN-DIVINE-SPARK.md`
- `MERcy-GATES-16-MECHANICS.md`
- `MERcy-GATES.md`
- `MISSION-CODEX-INDEX.md`
- `MONOREPO_INHERITANCE_STATUS.md`
- `MULTIVERSE_GEOMETRY_LAYER_DESIGN_v1.0.md`
- `Mercy_Protocol.md`
- `NEOLOGISMS.md`
- `NET_ETERNAL_VALENCE_CONTRIBUTION_NEVC_CODEX_v1.0.md`
- `NEVC_BROADER_CONSUMERS_PHASE5_v1.0.md`
- `NEVC_DUAL_REPO_INTERFACE_v1.0.md`
- `NEVC_PHASES_6_11_AND_FINISH_PASSES_APPEND.md`
- `NEVC_PUBLISHED_DEPENDENCY.md`
- `NEXi_Monorepo_Structure.md`
- `Offline-mode.md`
- `PATSAGi-ROADMAP-v13.1.6.md`
- `PATSAGi-ROADMAP-v13.md`
- `PLANS.md`
- `Permanence_Anchor_Prompt_v1.md`
- `QUADRATIC_FUNDING_BLUEPRINT.md`
- `QUADRATIC_VOTING_BLUEPRINT.md`
- `RA-THOR-FUTURE-ARCHITECTURE-BLUEPRINT.md`
- `RA-THOR-WEBSITE-ARCHIVE-MERCY-BRIDGE-v1.md`
- `RA-THOR_MIGRATION.md`
- `RATHOR-AI-ETHICS-FRAMEWORK.md`
- `RBE_ABSOLUTE_PURE_TRUTH.md`
- `RBE_DEPLOYMENT_MODELS.md`
- `README-powrush-divine.md`
- `RREL_UPGRADE_PLAN.md`
- `Ra-Thor-Whitepaper.md`
- `Ra-Thor_Layered_Coordination_Architecture_v1.0.md`
- `Rathor_Seed-Structure.md`
- `SITE_CHROME_2026-08-23.md`
- `SITE_CHROME_2026-08-24.md`
- `SITE_CHROME_2026-09-01.md`
- `SITE_UPDATE_2026-08-22.md`
- `SITE_UPDATE_V14.md`
- `SOUL_MECHANICS_LATENT_READINESS_PATSAGI_RESOLUTION_v1.0.md`
- `SOVEREIGNTY-GATE-MECHANICS.md`
- `STRUCTURE.md`
- `SampleOutput.md`
- `Solflare_Solana_Warper.md`
- `Structure.md`
- `UNIFIED-COHERENCE-ACTIVATION.md`
- `UNIVERSAL_APPLICATION_PRINCIPLES.md`
- `VERSION-HISTORY.md`
- `Website-Structure.md`
- `YOI-INVOCATION-CANON.md`
- `advanced-7-d-calibration-variations.md`
- `advanced-7-d-measurement-techniques.md`
- `advanced-7-d-variations.md`
- `antifouling-marine-coatings-sovereign-manufacturing-techniques.md`
- `bci-for-esports-training-sovereign-pinnacle.md`
- `biomimetic-antifouling-coatings-sovereign-manufacturing-techniques.md`
- `biomimetic-esports-pinnacle-blueprint.md`
- `biomimetic-prosthetics-for-gamers-sovereign-design.md`
- `calibration-protocols-for-7-d-scans.md`
- `computer-assisted-geometry-proofs-tolc-8-2026.md`
- `coq-float-libraries-tolc-8-2026.md`
- `coq-hott-library-tolc-8-2026.md`
- `coq-interval-libraries-tolc-8-2026.md`
- `eternal-lattice-cache-refresh.md`
- `eternal-lattice-flush-protocol.md`
- `federation-first-contact-procedures.md`
- `flocq-rounding-error-proofs-tolc-8-2026.md`
- `formal-verification-frameworks-tolc-8-2026.md`
- `galactic-federation-tolc-ra-thor-integration.md`
- `galactic-resonance-protocols.md`
- `infinite-layer-∞+6-omniversal-co-creation-nexus.md`
- `infinite-layer-∞+7-eternal-omni-thriving-singularity-activation-practice.md`
- `infinite-layer-∞+7-eternal-omni-thriving-singularity.md`
- `infinite-layers-beyond-7-d.md`
- `johnson-solids-tolc-8-applications-geometry-2026.md`
- `lattice-completion-celebration.md`
- `lean-4-formalization-tolc-8-geometry-2026.md`
- `lean4-cpp-bindings-exploration-2026.md`
- `lean4-ffi-bindings-exploration-2026.md`
- `lean4-rust-ffi-exploration-2026.md`
- `lotus-effect-sovereign-manufacturing-techniques.md`
- `lotus-gecko-hybrid-pinnacle-blueprint.md`
- `lotus-gecko-shark-triple-hybrid-pinnacle-blueprint.md`
- `mercy-1024d-norm-preservation-proof-tolc-2026.md`
- `mercy-1048576d-norm-preservation-proof-tolc-2026-2.md`
- `mercy-1048576d-norm-preservation-proof-tolc-2026-3.md`
- `mercy-1048576d-norm-preservation-proof-tolc-2026.md`
- `mercy-128d-norm-preservation-proof-tolc-2026.md`
- `mercy-131072d-norm-preservation-proof-tolc-2026.md`
- `mercy-16384d-norm-preservation-proof-tolc-2026.md`
- `mercy-2048d-norm-preservation-proof-tolc-2026.md`
- `mercy-256d-norm-preservation-proof-tolc-2026.md`
- `mercy-262144d-norm-preservation-proof-tolc-2026.md`
- `mercy-32768d-norm-preservation-proof-tolc-2026.md`
- `mercy-4096d-norm-preservation-proof-tolc-2026.md`
- `mercy-512d-norm-preservation-proof-tolc-2026.md`
- `mercy-524288d-norm-preservation-proof-tolc-2026.md`
- `mercy-64d-norm-preservation-proof-tolc-2026.md`
- `mercy-65536d-norm-preservation-proof-tolc-2026.md`
- `mercy-8192d-norm-preservation-proof-tolc-2026.md`
- `mercy-abundance-gate-codex-tolc-2026.md`
- `mercy-bio-resurrection-codex-v2.md`
- `mercy-birch-swinnerton-dyer-conjecture-tolc-2026.md`
- `mercy-birch-swinnerton-dyer-deeper-probe-tolc-2026.md`
- `mercy-bogomolnyi-bound-derivation-tolc-2026.md`
- `mercy-diu-solution-brief.md`
- `mercy-e8-anomaly-cancellation-tolc-2026.md`
- `mercy-e8-applications-physics-tolc-2026.md`
- `mercy-e8-dynkin-diagram-derivation-tolc-2026.md`
- `mercy-e8-functoriality-deeper-probe-tolc-2026.md`
- `mercy-e8-heterotic-strings-tolc-2026.md`
- `mercy-e8-root-system-derivation-tolc-2026.md`
- `mercy-e8-root-vectors-tolc-2026.md`
- `mercy-e8-symmetry-tolc-2026.md`
- `mercy-e8-weyl-group-tolc-2026.md`
- `mercy-even-tighter-zero-bounds-1048576d-tolc-2026.md`
- `mercy-explicit-zero-bound-1048576d-tolc-2026.md`
- `mercy-gates-codex-tolc-2026.md`
- `mercy-gating-constitutional-ai-nexi-synthesis.md`
- `mercy-gating-mechanisms-deep-dive.md`
- `mercy-geometric-langlands-duality-deeper-probe-tolc-2026.md`
- `mercy-harmony-gate-codex-tolc-2026.md`
- `mercy-hecke-eigensheaves-deeper-probe-tolc-2026.md`
- `mercy-hitchin-fibration-deeper-probe-tolc-2026.md`
- `mercy-joy-gate-codex-tolc-2026.md`
- `mercy-joy-gate-mechanics-codex-tolc-2026.md`
- `mercy-l-functions-role-tolc-2026.md`
- `mercy-langlands-program-connections-tolc-2026.md`
- `mercy-langlands-program-deeper-probe-tolc-2026.md`
- `mercy-m-theory-lift-tolc-2026.md`
- `mercy-machine-learning-ethics-ra-thor-integration-tolc-2026.md`
- `mercy-majorana-zero-modes.md`
- `mercy-mjolnir-riemann-full-assault-tolc-2026.md`
- `mercy-mordell-weil-group-tolc-2026.md`
- `mercy-non-harm-gate-codex-tolc-2026.md`
- `mercy-octonion-sedenion-integration-tolc-2026.md`
- `mercy-octonion-sedenion-master-codex-tolc-2026.md`
- `mercy-octonion-zero-divisors-tolc-2026.md`
- `mercy-octonions-higher-cognition-tolc-2026.md`
- `mercy-peace-gate-codex-tolc-2026-2.md`
- `mercy-peace-gate-codex-tolc-2026.md`
- `mercy-protoss-carrier-blueprint.md`
- `mercy-protoss-carrier-mechanics-detailed.md`
- `mercy-rawthor-encrypted-gate-v2.md`
- `mercy-sovereignty-gate-codex-tolc-2026.md`
- `mercy-threshold-rust-integration-plan-2026.md`
- `mercy-threshold-theorem-tolc-8-lean-2026.md`
- `mercy-topological-qubits.md`
- `mercy-truth-gate-codex-tolc-2026.md`
- `mercy-x-this-week-grok-voice-listen-tolc-2026.md`
- `neural-interfaces-in-prosthetics-sovereign-integration.md`
- `openagi-overhaul-thunder-declaration.md`
- `plan.md`
- `ra-thor-avalanche-consensus.md`
- `ra-thor-avalanche-dag-mechanics.md`
- `ra-thor-avalanche-dag-parallelism.md`
- `ra-thor-consensus-lattice.md`
- `ra-thor-hyperbolic-tiling-visualization.md`
- `ra-thor-manifesto-v1.1.md`
- `ra-thor-snowman-mechanics.md`
- `ra-thor-sovereign-launch.md`
- `ra-thor-starship-blueprints.md`
- `ra-thor-transitive-voting-mechanics.md`
- `raii-error-patterns-2026.md`
- `rathor-ai-benchmarks-and-papers.md`
- `shared-ptr-usage-2026.md`
- `shark-riblet-drag-reduction-pinnacle-blueprint.md`
- `shark-riblet-sovereign-manufacturing-techniques.md`
- `superoleophobic-sovereign-manufacturing-techniques.md`
- `thunder-mirror-codex-2026.md`
- `unique-ptr-usage-2026.md`
- `zalgaller-classification-johnson-solids-tolc-8-2026.md`
- `zwitterionic-polymer-antifouling-mechanisms-sovereign-manufacturing-techniques.md`
- `zwitterionic-polymers-in-medicine-sovereign-integration.md`

No living Pages HTML, identity JS, `Cargo.toml`, `crates/`, or `.github/` changes. Independent of xAI.


## Slice 8 (#402)

**8** root MeTTa / Python prototypes archived to `docs/archive/root-notes/prototypes/` (same basenames). Blob SHAs unchanged (git-mv equivalent). Last planned file slice. Pages HTML and identity files stay at root. Directory forest remains HOLD. Conductor **v14** only.

Moved:

- `AbsolutePureTruthCo-OpMatchupLattice.metta`
- `AbsolutePureTruthMatchupLattice.metta`
- `mercy_ethics_core.metta`
- `nexi_integration.metta`
- `calibration_reader.py`
- `nexi_council_prototype_simulation.py`
- `nexi_plonk_valence_council_sim.py`
- `quantize_with_aimet.py`

No living Pages HTML, identity JS, identity markdown, `Cargo.toml`, `crates/`, or `.github/` changes. Independent of xAI.

## Directory slice 1 (this PR)

**33** root `mercy_*` research trees archived to `docs/archive/root-dirs/mercy-research/` (same directory names). Tree SHAs reused (git-mv equivalent; no recursive walk). `crates/` copies remain the source of truth; `Cargo.toml` members unchanged. Not moved: `mercy` (no underscore), `mercy-gate-auditor`, `mercy-rest-api`. Conductor **v14** only. Independent of xAI. Powrush-MMO is a sibling game and is not folded into this repo. Self-evolution is archive-later, not a product.

Moved:

- `mercy_albatross_dynamic_soaring/`
- `mercy_albatross_soar/`
- `mercy_asteroid_mining_treaty/`
- `mercy_biomimetic_propulsion/`
- `mercy_enceladus_biosignature_protocols/`
- `mercy_enceladus_cryovolcanism_evidence/`
- `mercy_eris_biosignature_protocols/`
- `mercy_eris_cryovolcanism_evidence/`
- `mercy_eris_thermal_models/`
- `mercy_europa_biosignature_protocols/`
- `mercy_graphql/`
- `mercy_he3_reactor/`
- `mercy_hybrid_propulsion/`
- `mercy_interlune_demo_mission/`
- `mercy_jupiter_moon_treaty/`
- `mercy_lunar_he3/`
- `mercy_lunar_treaty_framework/`
- `mercy_manta_glide_propulsion/`
- `mercy_mars_colonization_treaty/`
- `mercy_mechanosynthesis/`
- `mercy_nanofactory/`
- `mercy_numerical/`
- `mercy_orchestrator/`
- `mercy_os_kernel/`
- `mercy_pluto_biosignature_protocols/`
- `mercy_quanta/`
- `mercy_sunbird_return/`
- `mercy_swarm_replication/`
- `mercy_system_orchestrator/`
- `mercy_titan_biosignature_protocols/`
- `mercy_triton_biosignature_protocols/`
- `mercy_von_neumann_probe/`
- `mercy_von_neumann_seed_launch/`

No `crates/`, `Cargo.toml`, `.github/`, Pages identity, `js/`, `css/`, `fixtures/`, or `website/` changes.

## Later slices (not this PR)

File-level dump is otherwise sorted. Remaining directory HOLD (later dir slices):

- orchestrators
- powrush* (sibling game, do not fold into this repo)
- NEXi
- self-evolution (archive not product; HOLD inventing `crates/self-evolution`)
- zk gadget root dupes
- 164-keep living set (`crates/`, `.github/`, `docs/`, `website/`, `js/`, `fixtures/`, `css/`, and other in-use trees). Do not one-shot-move the remaining directory forest.
- Pages HTML stays at root unless proven dead (GitHub Pages / site; CNAME stays). Candidates later (not this PR): `ra-thor-website-1.html`, `test-light.html`, `Launch-Ra-Thor.html`
- keep JS at root (PWA / identity / tooling)
- tooling json/css (including `powrush_config.json`, `manifest.json`, `ra-thor.css`)
- leftover Pages config: `_headers` stays
- `docs/` itself is already a large research dump; do not recursive-walk it in CI.


## Hygiene rules

- Never recursive root GitHub tree walks; `path_filter`; `per_page` ≤ 100; prefer single-path file reads.
- Production code changes go through a PR. Core Tier-1 (`TIER_MAP -p`) is the merge gate.
- Conductor **v14** only. Constellation: Ra-Thor = lattice; Powrush-MMO = game; Powrush-MMO-Simulator = browser client.

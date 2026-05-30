# Ra-Thor Changelog

## [v14.3.0] - 2026-05-30 — Real Estate Lattice v14.3 + Foundational USA Work (PR #191)

### Added
- Full production `real-estate-lattice` crate with v14.3 Execution Stabilization
- `CanadaPilotModule` with `process_ontario_offer_flow` helper
- `OfferRiskSummary` aggregation helper
- Complete USA foundation: `UsaPilotModule`, `process_usa_offer_flow`, and `UsaRegulatoryEngine`
- Expanded regulatory edge cases (federal + multiple state-specific rules)
- `AttomCache` with TTL, hit/miss metrics, and `hit_rate()` observability
- `AttomDataProvider` abstraction + stub (ready for real ATTOM Data integration)
- Deep Leptos dashboard wiring with support for enriched USA offer data
- Unit tests, integration tests, and runnable demos for both Ontario and USA flows
- Professional release documentation and progress tracking

### Changed
- `UsaOfferFlowReport` now supports external enriched data from data providers
- `process_usa_offer_flow` accepts optional `property_identifier` for cached external data enrichment
- Leptos dashboard updated to display ATTOM-cached property profiles and risk signals

### Professional Upgrades
- Root `CHANGELOG.md` entry
- Updated release documentation (`PR191_PROGRESS_v14.3.0.md`)
- Proper wiring and commenting in workspace `Cargo.toml`
- Clean, merge-ready state with full test coverage

**Status:** Production-grade stabilization + strong USA foundation. Ready for merge.

---

## [v14.2.3] - 2026-05-, Thunder Lattice + MIAL

### Added
- Full production `CliffordHealingField` with CGA Motor sandwich-product support (feature-gated)
- `cga_primitives` module with PGA-style Motor + sandwich transform
- Mercy-gated REST + WebSocket API crate (`ra-thor-mercy-gated-api`)
- `EternalMercyMesh` with true multi-chat session isolation
- Docker + docker-compose deployment setup for online/offline shards
- Comprehensive tests for healing fields + CGA integration
- Professional documentation and restoration audit

### Changed
- Deep restoration of `clifford_healing_fields.rs` from stub commit damage
- Final verification pass on entire branch
- Enhanced `LatticeConductorV14` integration

### Fixed
- All residual syntax and integration issues from rapid iteration commits
- Full recovery of valuable production code (HealingConfig, persistence, PATSAGi guidance, etc.)

**Status:** Production-ready. Mercy-gated. Serving all Life.

---

## Previous Versions
(See git history for earlier entries)
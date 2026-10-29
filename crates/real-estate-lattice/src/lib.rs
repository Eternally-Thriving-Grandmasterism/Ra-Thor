//! Design-only (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1 council note): not compiled; inherits workspace key powrush that the root does not define.
//! RREL-BUILD-1 amendment: non-compiling stubs gated behind `#[cfg(feature = "design-only")]`;
//! the always-on set compiles cleanly with standard workspace deps.
//! # Real Estate Lattice (RREL) — v14.15.0
//!
//! design-only, TOLC 8 checks planned (design-only, unverified), PATSAGi-aligned real estate organism.
//! Canada (Ontario/TREB/RECO) + USA pilot surfaces + Powrush land bridge.
//!
//! Living Cosmic Tick + ONE Organism readiness.
//! Contact: info@Rathor.ai

/// Canonical RREL surface version (Living Cosmic Tick aligned).
pub const RREL_VERSION: &str = "14.15.0";

// ── Always-on modules (clean dependency graph, compile by default) ─────────

// RECO AI Guidance compliance types — std-only, no external crate deps.
// Added as always-on by RREL-BUILD-1 (file authored in #730).
pub mod rrel_compliance;
pub use rrel_compliance::*;

// Core Ontario classifiers & form mapping
pub mod property_type_classifier;
pub mod deal_type_classifier;
pub mod form_mapping_engine;

// Offer validation (uses only classifiers above)
pub mod offer_package_validator;

// Multi-offer tracking (redis paths already cfg-gated inside the file)
pub mod multi_offer_track_engine;

// Risk & due diligence engines (no external crate deps)
pub mod developer_risk_engine;
pub mod status_certificate_analyzer;
pub mod lawyer_due_diligence_generator;
pub mod lawyer_report_generator;

// Ontario professional judgment
pub mod ontario_professional_judgment_layer;

// Composite deal readiness scoring (pure function, no crate deps)
pub mod deal_readiness_scoring;
pub use deal_readiness_scoring::calculate_deal_readiness_score;

// LAT / Divisional Court evidence packaging (serde + chrono only)
pub mod lat_divisional_court_evidence;

// ── RECO-relevant modules (RREL-UNGATE-1) ──────────────────────────────────
//
// These four modules were gated design-only in RREL-BUILD-1. RREL-UNGATE-1
// replaced their powrush/patsagi/mercy/quantum-swarm imports with local
// std-only stubs and fixed the ValidationResult↔OfferPackageValidation
// mismatch. They compile in the default build.

pub mod reco_form_handlers;
pub mod usa_mls_adapter;
pub mod quantum_real_estate_valuation;
pub mod offer_package_finalizer;

// ── Design-only stubs (STUB-AUDIT-1 WORKSPACE-KEY-SHELLS-1) ────────────────
//
// These modules carry the "design-only, unverified" label from their original
// authorship. They are NOT compiled in the default build because they require
// the powrush/patsagi-councils dependency tree, reference missing source files,
// or reference types not yet implemented (MercyEngine, mercy::traits, leptos,
// prometheus, …).
//
// Enable with `--features design-only` (requires resolving workspace-root
// conflict for powrush/patsagi-councils before any gated module can link).
//
// Other gated modules:
//   offer_package / compliance_helpers / disclosure_manager
//                          [no source file; only callers are design-only stubs]
//   offer_package_assembler
//                          [type mismatch; disclosure_manager dep]
//   rrel_brokerage_assembler, powrush_rrel_bridge, reco_form_handlers,
//   rrel_leptos_dashboard, rrel_lattice_conductor_bridge,
//   rrel_quantum_swarm_participant, rrel_tolc8_verifier,
//   rrel_tolc8_quantum_swarm_safety, rrel_tolc8_runtime_verifier
//                          [use patsagi_councils, mercy::traits, or leptos]
//   geometric_harmony_advisor, powrush_land_evaluation_bridge
//                          [use ra-thor-quantum-swarm-orchestrator (optional dep)]
//   usa_pilot_module       [incomplete source fragment; references design-only types]
//   usa_regulatory_engine, usa_state_adapters, usa_attom_cache,
//   usa_attom_data_provider, usa_mls_adapter (RECO-relevant), california_mls_adapter,
//   florida_mls_adapter, texas_mls_adapter, new_york_mls_adapter,
//   new_jersey_mls_adapter
//                          [use powrush, patsagi_councils, or MercyEngine]
//   canada_pilot_module    [uses patsagi_councils, MercyEngine, QuantumSwarmOrchestrator]
//   pms_bridge, pms_integration_bridge_engine
//                          [use powrush, patsagi_councils, MercyEngine]
//   mls_integration        [uses patsagi_councils]
//   reco_enforcement       [uses patsagi_councils]
//   observability          [uses prometheus/opentelemetry (not in deps)]
//   rrel_offer_risk_summary [references canada_pilot_module types]
//   avm_cache_invalidation, avm_invalidation_consumer, valuation_confidence_scorer
//                          [valuation_confidence_scorer is a placeholder stub]
//   quantum_real_estate_valuation (RECO-relevant), predictive_market_valuation_engine,
//   tax_optimization_engine, sustainable_property_upgrades_engine,
//   climate_resilience_engine, fintracker_integration_engine,
//   fractional_ownership, fractional_ownership_engine,
//   lease_renewal, lease_renewal_engine,
//   eviction_prevention, eviction_prevention_engine,
//   property_tax_appeal, property_tax_appeal_engine,
//   rental_income_distribution, rental_income_distribution_engine,
//   insurance_claim_engine, insurance_claim_risk_mitigation_engine,
//   maintenance_engine, multi_generational_wealth_transfer_engine,
//   emergency_response_disaster_recovery_engine,
//   tenant_screening, tenant_screening_engine, tenant_rights_habitability_engine,
//   community_land_trust_engine, landlord_compliance_reco_engine,
//   portfolio_optimization_engine, mortgage_engine, mortgage_processing_engine
//                          [use powrush, patsagi_councils, or MercyEngine]
//   space (submodule directory)
//                          [speculative research; uses mercy_radiation_shield etc.]

// Missing-file stubs (callers are all design-only)
#[cfg(feature = "design-only")]
pub mod offer_package;
#[cfg(feature = "design-only")]
pub mod compliance_helpers;
#[cfg(feature = "design-only")]
pub mod disclosure_manager;

// Offer lifecycle
#[cfg(feature = "design-only")]
pub mod offer_package_assembler;
// offer_package_finalizer moved to always-on above (RREL-UNGATE-1)

// Bridge / orchestration stubs
#[cfg(feature = "design-only")]
pub mod rrel_brokerage_assembler;
#[cfg(feature = "design-only")]
pub mod powrush_rrel_bridge;
// reco_form_handlers moved to always-on above (RREL-UNGATE-1)
#[cfg(feature = "design-only")]
pub mod rrel_lattice_conductor_bridge;
#[cfg(feature = "design-only")]
pub mod rrel_quantum_swarm_participant;
#[cfg(feature = "design-only")]
pub mod rrel_leptos_dashboard;

// TOLC8 verifier stubs
#[cfg(feature = "design-only")]
pub mod rrel_tolc8_verifier;
#[cfg(feature = "design-only")]
pub mod rrel_tolc8_quantum_swarm_safety;
#[cfg(feature = "design-only")]
pub mod rrel_tolc8_runtime_verifier;

// Geometric harmony (needs optional quantum-swarm-orchestrator)
#[cfg(feature = "design-only")]
pub mod geometric_harmony_advisor;
#[cfg(feature = "design-only")]
pub mod powrush_land_evaluation_bridge;

// USA modules
#[cfg(feature = "design-only")]
pub mod usa_pilot_module;
#[cfg(feature = "design-only")]
pub mod usa_regulatory_engine;
#[cfg(feature = "design-only")]
pub mod usa_state_adapters;
#[cfg(feature = "design-only")]
pub mod usa_attom_cache;
#[cfg(feature = "design-only")]
pub mod usa_attom_data_provider;
// usa_mls_adapter moved to always-on above (RREL-UNGATE-1)
#[cfg(feature = "design-only")]
pub mod california_mls_adapter;
#[cfg(feature = "design-only")]
pub mod florida_mls_adapter;
#[cfg(feature = "design-only")]
pub mod texas_mls_adapter;
#[cfg(feature = "design-only")]
pub mod new_york_mls_adapter;
#[cfg(feature = "design-only")]
pub mod new_jersey_mls_adapter;

// Canada pilot
#[cfg(feature = "design-only")]
pub mod canada_pilot_module;

// PMS / brokerage bridges
#[cfg(feature = "design-only")]
pub mod pms_bridge;
#[cfg(feature = "design-only")]
pub mod pms_integration_bridge_engine;

// MLS / RECO
#[cfg(feature = "design-only")]
pub mod mls_integration;
#[cfg(feature = "design-only")]
pub mod reco_enforcement;

// Observability (prometheus/otel not in deps)
#[cfg(feature = "design-only")]
pub mod observability;

// Risk summary (references canada_pilot_module types)
#[cfg(feature = "design-only")]
pub mod rrel_offer_risk_summary;

// AVM cache chain (valuation_confidence_scorer is a placeholder)
#[cfg(feature = "design-only")]
pub mod valuation_confidence_scorer;
#[cfg(feature = "design-only")]
pub mod avm_cache_invalidation;
#[cfg(feature = "design-only")]
pub mod avm_invalidation_consumer;

// Additional domain stubs
// quantum_real_estate_valuation moved to always-on above (RREL-UNGATE-1)
#[cfg(feature = "design-only")]
pub mod predictive_market_valuation_engine;
#[cfg(feature = "design-only")]
pub mod tax_optimization_engine;
#[cfg(feature = "design-only")]
pub mod sustainable_property_upgrades_engine;
#[cfg(feature = "design-only")]
pub mod climate_resilience_engine;
#[cfg(feature = "design-only")]
pub mod fintracker_integration_engine;
#[cfg(feature = "design-only")]
pub mod fractional_ownership;
#[cfg(feature = "design-only")]
pub mod fractional_ownership_engine;
#[cfg(feature = "design-only")]
pub mod lease_renewal;
#[cfg(feature = "design-only")]
pub mod lease_renewal_engine;
#[cfg(feature = "design-only")]
pub mod eviction_prevention;
#[cfg(feature = "design-only")]
pub mod eviction_prevention_engine;
#[cfg(feature = "design-only")]
pub mod property_tax_appeal;
#[cfg(feature = "design-only")]
pub mod property_tax_appeal_engine;
#[cfg(feature = "design-only")]
pub mod rental_income_distribution;
#[cfg(feature = "design-only")]
pub mod rental_income_distribution_engine;
#[cfg(feature = "design-only")]
pub mod insurance_claim_engine;
#[cfg(feature = "design-only")]
pub mod insurance_claim_risk_mitigation_engine;
#[cfg(feature = "design-only")]
pub mod maintenance_engine;
#[cfg(feature = "design-only")]
pub mod multi_generational_wealth_transfer_engine;
#[cfg(feature = "design-only")]
pub mod emergency_response_disaster_recovery_engine;
#[cfg(feature = "design-only")]
pub mod tenant_screening;
#[cfg(feature = "design-only")]
pub mod tenant_screening_engine;
#[cfg(feature = "design-only")]
pub mod tenant_rights_habitability_engine;
#[cfg(feature = "design-only")]
pub mod community_land_trust_engine;
#[cfg(feature = "design-only")]
pub mod landlord_compliance_reco_engine;
#[cfg(feature = "design-only")]
pub mod portfolio_optimization_engine;
#[cfg(feature = "design-only")]
pub mod mortgage_engine;
#[cfg(feature = "design-only")]
pub mod mortgage_processing_engine;

// Space / speculative research (uses mercy_radiation_shield etc.)
#[cfg(feature = "design-only")]
pub mod space;

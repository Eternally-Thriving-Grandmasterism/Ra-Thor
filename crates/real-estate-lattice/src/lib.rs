//! Ra-Thor Real Estate Lattice (RREL) — v0.5.21
//! Mercy-Gated Global Real Estate Operating System
//! Canada Pilot (v0.5.19) + USA Expansion (v0.5.21) + Full Integration

pub mod pms_bridge;
pub mod mls_integration;
pub mod reco_enforcement;
pub mod lat_divisional_court_evidence;
pub mod quantum_real_estate_valuation;
pub mod canada_pilot_module;

// === USA EXPANSION (v0.5.21) ===
pub mod usa_regulatory_engine;
pub mod usa_mls_adapter;
pub mod california_mls_adapter;

pub use crate::pms_bridge::{PmsBridge, PmsProvider, RrelError};
pub use crate::mls_integration::{TrebMlsAdapter, MlsListing, MlsError};
pub use crate::reco_enforcement::{RecoEnforcementEngine, RecoEnforcementAction, RecoError};
pub use crate::lat_divisional_court_evidence::{EvidenceGenerator, LatAppealPackage, DivisionalCourtPackage};
pub use crate::quantum_real_estate_valuation::{QuantumRealEstateValuation, ValuationResult};
pub use crate::canada_pilot_module::{CanadaPilotModule, CanadaPilotReport};

pub use crate::usa_regulatory_engine::{UsaRegulatoryEngine, UsaRegulatoryResult, UsaRegulatoryError};
pub use crate::usa_mls_adapter::{UsaMlsAdapter, UsaListing, UsaMlsError};
pub use crate::california_mls_adapter::{CaliforniaMlsAdapter, CaliforniaMlsError};

pub const RREL_VERSION: &str = "0.5.21";

pub mod prelude {
    pub use super::*;
}

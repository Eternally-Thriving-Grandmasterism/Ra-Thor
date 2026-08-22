//! # Mercy-Security — White-Hat AGSi Defense (v14.15.5)
//!
//! Domain profiles: education · research · enterprise · creative · robotics · biomedical
//! Physical actuation + wet-lab synthesis hard-refuse under HarmRefusalPolicy.
//! Medium+ ingestion blocks feed UnifiedAgentSurface isolation + fleet signals.
//! TOLC 8 + PATSAGi | AG-SML v1.0 | Contact: info@Rathor.ai

mod domain_profiles;
mod safe_agent_runtime;
mod mercy_council_fleet;
mod unified_agent_surface;
pub mod agsi_eval;

pub use safe_agent_runtime::{
    AgentActionReceipt, AgentActionRequest, SafeAgentRuntime, AGENT_TOKEN_MAX_TTL_SECS,
};
pub use mercy_council_fleet::{
    AgentIsolationLevel, FleetAgentSlot, FleetRiskTier, FleetSecuritySignal, MercyCouncilFleet,
    DEFAULT_PER_AGENT_BUDGET_SHARE, FLEET_PROGRESSIVE_VALENCE_FLOOR,
};
pub use unified_agent_surface::{
    UnifiedAgentSurface, WhitehatIngestionOutcome, GOVERNOR_TRIPS_PER_ISOLATION_STEP,
};
pub use domain_profiles::{AuditChainStep, ClassroomAuditReport};

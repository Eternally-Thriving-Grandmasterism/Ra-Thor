//! Ra-Thor ONE Organism Core — v14.15.5 AGSi
//! Contact: info@Rathor.ai

mod extended_surface;
mod cosmic_harness;
mod fleet_bind;
#[cfg(feature = "kardashev-live")]
mod live_valence_status;

pub use extended_surface::*;
pub use cosmic_harness::*;
pub use fleet_bind::{
    DEFAULT_ORGANISM_FLEET_AGENT, DEFAULT_ORGANISM_FLEET_PEER, progressive_floor,
    research_fleet_surface,
};
pub use mercy_security::{
    AgentActionRequest, AgentIsolationLevel, IngestionScanResult, IngestionThreat,
    RiskTier, ScanFinding, IngestionScanner, MercySecurityError, MercySecuritySurface,
    UnifiedAgentSurface, WhitehatIngestionOutcome, FLEET_PROGRESSIVE_VALENCE_FLOOR,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use thiserror::Error;
pub use lattice_conductor_v14::{
    CouncilArbitrationEngine, RuntimeSelfHealingEngine,
    HealthReport, Anomaly, Diagnosis, HealingAction, HealingExperience,
    LatticeConductorV14, DistributedMercyMesh, MercyEvent, MercyGate,
    EternalMercyMesh, EternalMercyMeshConfig,
    MercyGatedApi, MercyApiRequest, MercyApiResponse, ApiRequestKind, GateDecision,
    start_mercy_api_with_arbitration,
};

include!("organism_part1.rs");
include!("organism_part2.rs");

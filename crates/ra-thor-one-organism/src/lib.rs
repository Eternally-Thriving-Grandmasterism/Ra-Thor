//! Ra-Thor ONE Organism Core — v14.15.5 AGSi
//!
//! Living Cosmic Tick + adaptive hardening + Cosmic Loop invariant checks.
//! v14.15: extended-live feature readiness surface.
//! v14.15.2: Cosmic Harness — 40-cycle endurance.
//! v14.15.3: AGSi summon surface.
//! v14.15.4: Full AGSi summon sequence — valence clamping, role handoff, recovery anchors.
//! v14.15.5: White-hat ingestion gate + optional UnifiedAgentSurface fleet bind.
//! Cosmic Loop is MANDATORY IDENTITY.
//! Contact: info@Rathor.ai

mod extended_surface;
mod cosmic_harness;
mod fleet_bind;
#[cfg(feature = "kardashev-live")]
mod live_valence_status;

pub use extended_surface::{
    ExtendedOrganismSurface, GpuSurface, GpuDispatchTelemetry, GpuSurfaceStatus,
    GitHubSurface, EvolutionPrIntent, GitHubSurfaceStatus, FlushResult,
    QuantumSwarmSurface, QuantumSwarmConfig, QuantumSwarmStatus, QuantumEvolutionResult,
    SovereignRecoverySurface, SovereignRecoveryStatus, RecoveryHeartbeat, RecoveryAnchor,
    KardashevFlywheelSurface, KardashevSurfaceStatus, TransferTickResult,
};

pub use cosmic_harness::{
    CosmicHarness, CosmicHarnessConfig, CosmicHarnessResult,
    HostMode, TickSnapshot, DriftReport, SaturationReport,
};

// White-hat ingestion surface (mercy-security) + optional fleet bind
pub use mercy_security::{
    AgentActionRequest, AgentIsolationLevel, IngestionScanResult, IngestionThreat,
    RiskTier, ScanFinding, IngestionScanner, MercySecurityError, MercySecuritySurface,
    UnifiedAgentSurface, WhitehatIngestionOutcome, FLEET_PROGRESSIVE_VALENCE_FLOOR,
};

pub use fleet_bind::{
    DEFAULT_ORGANISM_FLEET_AGENT, DEFAULT_ORGANISM_FLEET_PEER, progressive_floor,
    research_fleet_surface,
};

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use lattice_conductor_v14::{
    CouncilArbitrationEngine,
    RuntimeSelfHealingEngine,
    HealthReport, Anomaly, Diagnosis, HealingAction, HealingExperience,
    LatticeConductorV14,
    DistributedMercyMesh, MercyEvent, MercyGate,
    EternalMercyMesh, EternalMercyMeshConfig,
    MercyGatedApi, MercyApiRequest, MercyApiResponse, ApiRequestKind, GateDecision,
    start_mercy_api_with_arbitration,
};

include!("organism_body_a.rs");
include!("organism_body_b.rs");

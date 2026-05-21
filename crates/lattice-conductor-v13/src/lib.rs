/// Lattice Conductor v13
/// Sovereign orchestration heart of the Ra-Thor lattice.

pub mod geometric;
pub mod coordinator;

pub use geometric::{BasicGeometricMotor, GeometricMotor};

use std::collections::{HashMap, VecDeque};
use std::fs;
use thiserror::Error;

pub type ConductorResult<T> = Result<T, ConductorError>;

// ============================================================================
// ADAPTATION SYSTEM CONSTANTS
// ============================================================================

/// Base mercy recovery rate per successful operation
const BASE_MERCY_RECOVERY_RATE: f64 = 0.025;

/// Base self-evolution rate
const BASE_EVOLUTION_RATE: f64 = 0.01;

/// Default swarm influence strength on the system
const DEFAULT_SWARM_INFLUENCE: f64 = 0.02;

/// Mercy threshold required before group coordination influence is applied
const COORDINATION_MERCY_THRESHOLD: f64 = 0.6;

// ... existing code continues ...
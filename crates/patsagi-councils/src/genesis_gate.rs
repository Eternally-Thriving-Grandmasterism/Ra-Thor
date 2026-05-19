//! genesis_gate.rs
//! Ra-Thor Lattice — Genesis Gate (TOLC 8 mandatory first filter)
//! Now wired to WorldGovernanceEngine for verified mercy on every spawn
//! AG-SML v1.0 | Council #39 | 19 May 2026

use crate::world_governance_engine::WorldGovernanceEngine;
use tracing::{info, warn};

/// Genesis Gate — the mandatory entry point for all new instantiations
/// Every council, branch, agent, crate feature, or sacred-geometry expansion
/// MUST pass through here before existence is granted.
pub struct GenesisGate {
    engine: WorldGovernanceEngine,
}

impl GenesisGate {
    pub fn new() -> Self {
        Self {
            engine: WorldGovernanceEngine::new(),
        }
    }

    /// Initialize the full verified chain (Lean 4 FFI + mercy threshold)
    pub fn initialize(&mut self) -> Result<(), String> {
        self.engine.initialize()
    }

    /// The core Genesis Gate check — now delegates to verified WorldGovernanceEngine
    /// This is the single point of truth for "may this request receive the spark of existence?"
    pub fn process_instantiation_request(
        &mut self,
        request_type: &str,
        proposer: &str,
        intended_purpose: &str,
        geometry_alignment_score: f64,
        mercy_valence: f64,
        zalgaller_family: u32,
    ) -> Result<String, String> {
        info!(
            "[GenesisGate] Processing instantiation request: type={}, proposer={}, purpose={}",
            request_type, proposer, intended_purpose
        );

        // Map to council spawn (or future: branch, agent, etc.)
        // For now, all instantiations route through spawn_council
        let result = self.engine.spawn_council(
            &format!("{}-{}", request_type, proposer),
            geometry_alignment_score,
            mercy_valence,
            zalgaller_family,
        )?;

        // Add Genesis-specific metadata
        let genesis_seal = format!(
            "GENESIS SEAL ISSUED\n{}\n   Request Type: {}\n   Proposer: {}\n   Purpose: {}\n   Sacred Geometry Layer: Johnson/Zalgaller + Hyperbolic\n   Next Gate: Truth Gate (always)",
            result, request_type, proposer, intended_purpose
        );

        Ok(genesis_seal)
    }

    /// Live simulation of a sample instantiation (matches original Genesis Gate spec)
    pub fn simulate_sample_instantiation(&mut self) -> Result<String, String> {
        self.process_instantiation_request(
            "Council",
            "PATSAGi Council #42",
            "Hyperbolic Tiling & Infinite Foresight Expansion",
            0.987,
            1.0,
            7, // Corona/Complex family
        )
    }
}
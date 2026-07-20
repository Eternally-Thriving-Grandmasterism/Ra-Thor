//! Genesis Gate — v14.15.0
//!
//! TOLC 8 mandatory first filter for all new instantiations.
//! Routes through WorldGovernanceEngine for verified mercy on every spawn.
//!
//! Living Cosmic Tick aligned.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::world_governance_engine::WorldGovernanceEngine;
use tracing::info;

/// Genesis Gate — mandatory entry point for all new instantiations.
/// Every council, branch, agent, or sacred-geometry expansion must pass here.
pub struct GenesisGate {
    engine: WorldGovernanceEngine,
    seals_issued: u64,
}

impl GenesisGate {
    pub fn new() -> Self {
        Self {
            engine: WorldGovernanceEngine::new(),
            seals_issued: 0,
        }
    }

    /// Initialize the verified chain (Lean 4 FFI + mercy threshold when enabled).
    pub fn initialize(&mut self) -> Result<(), String> {
        self.engine.initialize()
    }

    /// Core Genesis Gate check — single point of truth for instantiation.
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
            "[GenesisGate v14.15.0] type={} proposer={} purpose={}",
            request_type, proposer, intended_purpose
        );

        let result = self.engine.spawn_council(
            &format!("{}-{}", request_type, proposer),
            geometry_alignment_score,
            mercy_valence,
            zalgaller_family,
        )?;

        self.seals_issued = self.seals_issued.saturating_add(1);

        Ok(format!(
            "GENESIS SEAL ISSUED (v14.15.0)\n{}\n   Request Type: {}\n   Proposer: {}\n   Purpose: {}\n   Seals issued: {}\n   Sacred Geometry: Johnson/Zalgaller + Hyperbolic\n   Next Gate: Truth Gate\n   Living Cosmic Tick: active",
            result, request_type, proposer, intended_purpose, self.seals_issued
        ))
    }

    /// Sample instantiation for demos / tests.
    pub fn simulate_sample_instantiation(&mut self) -> Result<String, String> {
        self.process_instantiation_request(
            "Council",
            "PATSAGi Council #42",
            "Hyperbolic Tiling & Infinite Foresight Expansion",
            0.987,
            1.0,
            7,
        )
    }

    pub fn summary(&self) -> String {
        format!(
            "GenesisGate v14.15.0 | seals_issued={} | {}",
            self.seals_issued,
            self.engine.summary()
        )
    }
}

impl Default for GenesisGate {
    fn default() -> Self {
        Self::new()
    }
}

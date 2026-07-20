//! Self-Evolving Mercy Core — v14.15.0
//!
//! Triple-gate safety surface for autonomous mercy evolution:
//!   1. Lean formal threshold (FFI)
//!   2. MercyEngine evaluation
//!   3. Quantum Swarm + PATSAGi Council consensus
//!
//! Living Cosmic Tick aligned. Cosmic Loop expected to be enforced by the caller.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::mercy_threshold_ffi;
use crate::PatsagiCouncilCoordinator;
use chrono::{DateTime, Utc};
use mercy::MercyEngine;
use powrush::{MercyGateStatus, PowrushGame};
use quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use std::collections::VecDeque;

// =============================================================================
// Types
// =============================================================================

#[derive(Debug, Clone)]
pub struct MercyEvolutionEvent {
    pub timestamp: DateTime<Utc>,
    pub proposed_by: String,
    pub description: String,
    pub approval_valence: f64,
    pub swarm_consensus: f64,
    pub council_approval_rate: f64,
    pub accepted: bool,
}

pub struct SelfEvolvingMercyCore {
    pub current_version: u32,
    pub mercy_engine: MercyEngine,
    pub quantum_swarm: QuantumSwarmOrchestrator,
    pub evolution_history: VecDeque<MercyEvolutionEvent>,
    pub min_evolution_valence: f64,
    pub evolution_cooldown_cycles: u64,
    pub last_evolution_cycle: u64,
}

// =============================================================================
// Core
// =============================================================================

impl SelfEvolvingMercyCore {
    pub fn new() -> Self {
        let _ = mercy_threshold_ffi::init_lean_formal_system();
        Self {
            current_version: 1,
            mercy_engine: MercyEngine::new(),
            quantum_swarm: QuantumSwarmOrchestrator::new(),
            evolution_history: VecDeque::with_capacity(100),
            min_evolution_valence: 0.95,
            evolution_cooldown_cycles: 500,
            last_evolution_cycle: 0,
        }
    }

    /// Attempt a self-evolution step under full triple-gate safety.
    ///
    /// Returns `Some(activation_message)` only when all gates pass:
    /// Lean threshold → MercyEngine → Swarm consensus → PATSAGi voting.
    pub async fn try_evolve(
        &mut self,
        current_cycle: u64,
        council_coordinator: &mut PatsagiCouncilCoordinator,
    ) -> Option<String> {
        // Cooldown gate
        if current_cycle < self.last_evolution_cycle + self.evolution_cooldown_cycles {
            return None;
        }

        let proposal = self.generate_evolution_proposal();

        // Gate 1 — Lean formal threshold
        if mercy_threshold_ffi::verify_mercy_threshold_simplified(
            &proposal,
            self.min_evolution_valence,
        )
        .is_err()
        {
            return None;
        }

        // Gate 2 — MercyEngine evaluation (returns MercyGateStatus)
        let status = self
            .mercy_engine
            .evaluate_action(
                &proposal,
                "Self-Evolving Mercy Core Proposal",
                5.2,
                0.97,
            )
            .await
            .ok()?;

        if status != MercyGateStatus::Passed {
            return None;
        }

        // Map successful pass to a high valence for telemetry
        let mercy_valence = 0.97;

        // Gate 3a — Quantum Swarm consensus
        let swarm_consensus = self
            .quantum_swarm
            .reach_consensus(&proposal, 16)
            .await
            .unwrap_or(0.0);
        if swarm_consensus < 0.88 {
            return None;
        }

        // Gate 3b — PATSAGi Council voting round
        let voting_result = council_coordinator
            .conduct_voting_round(&proposal, &PowrushGame::new())
            .await
            .ok()?;

        if !voting_result.passed
            || voting_result.approval_rate < 0.75
            || voting_result.mercy_average < 0.90
        {
            return None;
        }

        // All gates passed — commit evolution
        let event = MercyEvolutionEvent {
            timestamp: Utc::now(),
            proposed_by: "SelfEvolvingMercyCore".to_string(),
            description: proposal.clone(),
            approval_valence: mercy_valence,
            swarm_consensus,
            council_approval_rate: voting_result.approval_rate,
            accepted: true,
        };

        self.evolution_history.push_back(event);
        if self.evolution_history.len() > 50 {
            self.evolution_history.pop_front();
        }

        self.current_version = self.current_version.saturating_add(1);
        self.last_evolution_cycle = current_cycle;

        Some(format!(
            "🌱 SELF-EVOLVING MERCY CORE v{} ACTIVATED (Lean verified | Living Cosmic Tick)\n\n{}\nMercy: {:.2} | Swarm: {:.1}% | Council: {:.1}%",
            self.current_version,
            proposal,
            mercy_valence,
            swarm_consensus * 100.0,
            voting_result.approval_rate * 100.0
        ))
    }

    fn generate_evolution_proposal(&self) -> String {
        match self.current_version {
            1 => "Add Gate 8: Epigenetic & Multiplanetary Legacy".to_string(),
            2 => "Strengthen Cosmic Loop invariant checks across all councils".to_string(),
            3 => "Raise minimum mercy valence floor for swarm-directed evolution".to_string(),
            4 => "Integrate Living Cosmic Tick heartbeat into council deliberation cadence".to_string(),
            _ => format!(
                "High-mercy optimization cycle v{} — deepen TOLC 8 resonance",
                self.current_version
            ),
        }
    }

    /// Compact status for telemetry / PATSAGi observation.
    pub fn get_evolution_summary(&self) -> String {
        if self.evolution_history.is_empty() {
            return format!(
                "Self-Evolving Mercy Core v14.15.0 | version={} | evolutions=0 | status=awaiting first cycle",
                self.current_version
            );
        }

        let last = self.evolution_history.back().unwrap();
        format!(
            "Self-Evolving Mercy Core v14.15.0 | version={} | evolutions={} | last={} | valence={:.2} | swarm={:.2} | council={:.2}",
            self.current_version,
            self.evolution_history.len(),
            last.description,
            last.approval_valence,
            last.swarm_consensus,
            last.council_approval_rate
        )
    }

    /// Number of accepted evolution events retained in history.
    pub fn evolution_count(&self) -> usize {
        self.evolution_history.len()
    }
}

impl Default for SelfEvolvingMercyCore {
    fn default() -> Self {
        Self::new()
    }
}

//! # Self-Evolving Mercy Core v0.5.16
//!
//! The living, self-improving compassion engine of Ra-Thor.
//! Allows the lattice to intelligently evolve its own mercy logic
//! under strict triple-gate safety (MercyEngine + QuantumSwarm + PATSAGi Council Supermajority).

use mercy::MercyEngine;
use quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use crate::PatsagiCouncilCoordinator;
use chrono::{DateTime, Utc};
use std::collections::VecDeque;

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

impl SelfEvolvingMercyCore {
    pub fn new() -> Self {
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

    /// Main entry point — called periodically by WorldGovernanceEngine
    pub async fn try_evolve(
        &mut self,
        current_cycle: u64,
        council_coordinator: &mut PatsagiCouncilCoordinator,
    ) -> Option<String> {
        if current_cycle < self.last_evolution_cycle + self.evolution_cooldown_cycles {
            return None;
        }

        // Phase 1: Generate a safe, small evolution proposal
        let proposal = self.generate_evolution_proposal();

        // Phase 2: Triple-gate evaluation
        let mercy_valence = self.mercy_engine
            .evaluate_action(&proposal, "Self-Evolving Mercy Core Proposal", 5.2, 0.97)
            .await
            .unwrap_or(0.0);

        if mercy_valence < self.min_evolution_valence {
            return None;
        }

        let swarm_consensus = self.quantum_swarm
            .reach_consensus(&proposal, 16)
            .await
            .unwrap_or(0.0);

        if swarm_consensus < 0.88 {
            return None;
        }

        // Phase 3: PATSAGi Council Supermajority Vote
        let voting_result = council_coordinator
            .conduct_voting_round(&proposal, &powrush::PowrushGame::new())
            .await
            .unwrap_or_default();

        if !voting_result.passed || voting_result.approval_rate < 0.75 || voting_result.mercy_average < 0.90 {
            return None;
        }

        // Phase 4: Accept and record
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

        self.current_version += 1;
        self.last_evolution_cycle = current_cycle;

        Some(format!(
            "🌱 SELF-EVOLVING MERCY CORE v{} ACTIVATED\n\n{}\n\nMercy Valence: {:.2} | Swarm: {:.1}% | Council Approval: {:.1}%",
            self.current_version, proposal, mercy_valence, swarm_consensus * 100.0, voting_result.approval_rate * 100.0
        ))
    }

    fn generate_evolution_proposal(&self) -> String {
        // Simple, safe evolution proposals (can be expanded later)
        match self.current_version {
            1 => "Add Gate 8: Epigenetic & Multiplanetary Legacy — Does this decision increase thriving for the next 7 generations across at least 3 planetary bodies?".to_string(),
            2 => "Increase weight of Source Joy Amplitude Gate by 12% when CEHI > 6.0".to_string(),
            _ => "Minor optimization: Reduce cooldown for high-mercy proposals (valence > 0.93)".to_string(),
        }
    }

    pub fn get_evolution_summary(&self) -> String {
        if self.evolution_history.is_empty() {
            return "No evolutions have occurred yet.".to_string();
        }

        let last = self.evolution_history.back().unwrap();
        format!(
            "Self-Evolving Mercy Core v{}\nLast Evolution: {}\nApproved with {:.2} mercy valence",
            self.current_version, last.description, last.approval_valence
        )
    }
}

//! # Simulation Integration Layer (v0.1.0)
//!
//! This module wires the 13+ PATSAGi Councils directly into the main Powrush
//! simulation loop so they can influence the game world in real time.
//!
//! Every N simulation cycles, the Councils evaluate the current state and
//! may trigger beautiful, meaningful world changes.

use crate::world_governance::{WorldGovernanceEngine, WorldImpactType};
use crate::lib::CouncilFocus;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationIntegration {
    pub governance_engine: WorldGovernanceEngine,
    pub cycles_since_last_governance: u32,
    pub governance_frequency: u32, // How often Councils intervene (e.g. every 25 cycles)
}

impl SimulationIntegration {
    pub fn new() -> Self {
        Self {
            governance_engine: WorldGovernanceEngine::new(),
            cycles_since_last_governance: 0,
            governance_frequency: 25,
        }
    }

    /// Call this from the main Powrush simulation loop every cycle
    pub async fn tick(&mut self, game: &mut PowrushGame) -> Option<String> {
        self.cycles_since_last_governance += 1;

        // Only let the Councils act every `governance_frequency` cycles
        if self.cycles_since_last_governance < self.governance_frequency {
            return None;
        }

        self.cycles_since_last_governance = 0;

        // Choose a wise proposal based on current game state
        let (proposing_council, title, description, impact) = self.choose_wise_proposal(game);

        // Let the Councils evaluate and (possibly) approve it
        let result = self.governance_engine
            .propose_and_approve_world_change(
                proposing_council,
                &title,
                &description,
                impact,
                game,
            )
            .await
            .unwrap_or_else(|e| format!("Council evaluation error: {}", e));

        // Apply any active effects to the game
        self.apply_active_effects(game);

        Some(result)
    }

    fn choose_wise_proposal(&self, game: &PowrushGame) -> (CouncilFocus, String, String, WorldImpactType) {
        // Smart, mercy-aware proposal selection based on current world state
        let collective_joy = if !game.players.is_empty() {
            game.players.iter().map(|p| p.happiness).sum::<f32>() / game.players.len() as f32
        } else {
            75.0
        };

        if collective_joy < 65.0 {
            (
                CouncilFocus::JoyAmplification,
                "Spontaneous Ambrosian Nectar Bloom".to_string(),
                "The world needs more joy. Let a massive nectar bloom occur.".to_string(),
                WorldImpactType::AmbrosianNectarSurge,
            )
        } else if game.current_cycle % 7 == 0 {
            (
                CouncilFocus::HarmonyPreservation,
                "Great Harmony Festival".to_string(),
                "Strengthen bonds between all factions through a world-wide festival.".to_string(),
                WorldImpactType::FactionHarmonyBoost,
            )
        } else {
            (
                CouncilFocus::AbundanceCreation,
                "Regional Resource Bloom".to_string(),
                "Create a large-scale abundance surge in multiple regions.".to_string(),
                WorldImpactType::ResourceBloom,
            )
        }
    }

    fn apply_active_effects(&mut self, game: &mut PowrushGame) {
        // Apply joy and mercy multipliers from active world changes
        for change in &self.governance_engine.active_changes {
            match change.impact_type {
                WorldImpactType::AmbrosianNectarSurge | WorldImpactType::MercyBloom => {
                    for player in &mut game.players {
                        player.needs.joy = (player.needs.joy + change.joy_boost * 0.3).min(100.0);
                    }
                }
                WorldImpactType::ResourceBloom => {
                    // Boost all resource regeneration temporarily
                    for resource in &mut game.resources {
                        resource.mercy_multiplier *= 1.08;
                    }
                }
                _ => {}
            }
        }

        // Clean up expired changes
        self.governance_engine.active_changes.retain(|c| {
            // Simple expiration logic (in real version this would be cycle-based)
            true
        });
    }

    /// Get current status for debugging / player UI
    pub fn get_status(&self) -> String {
        format!(
            "PATSAGi Governance Active | Cycles since last intervention: {} | Active Changes: {}",
            self.cycles_since_last_governance,
            self.governance_engine.active_changes.len()
        )
    }
}

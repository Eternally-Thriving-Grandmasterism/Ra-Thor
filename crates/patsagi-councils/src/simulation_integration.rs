//! # Simulation Integration Layer — v14.15.0
//!
//! Wires the 16 PATSAGi Councils into the main Powrush simulation loop
//! so they can influence the game world in real time.
//!
//! Every N simulation cycles, the Councils evaluate the current state and
//! may trigger meaningful world changes with mechanical impact.
//!
//! Living Cosmic Tick aligned.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::{CouncilFocus, WorldGovernanceEngine, WorldImpactType};
use powrush::PowrushGame;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationIntegration {
    pub governance_engine: WorldGovernanceEngine,
    pub cycles_since_last_governance: u32,
    /// How often Councils intervene (in simulation cycles).
    pub governance_frequency: u32,
    pub interventions: u64,
}

impl SimulationIntegration {
    pub fn new() -> Self {
        Self {
            governance_engine: WorldGovernanceEngine::new(),
            cycles_since_last_governance: 0,
            governance_frequency: 25,
            interventions: 0,
        }
    }

    /// Call this from the main Powrush simulation loop every cycle.
    pub async fn tick(&mut self, game: &mut PowrushGame) -> Option<String> {
        self.cycles_since_last_governance = self.cycles_since_last_governance.saturating_add(1);

        if self.cycles_since_last_governance < self.governance_frequency {
            return None;
        }

        self.cycles_since_last_governance = 0;
        self.interventions = self.interventions.saturating_add(1);

        let (proposing_council, title, description, impact) = self.choose_wise_proposal(game);

        let result = self
            .governance_engine
            .propose_and_approve_world_change(
                proposing_council,
                &title,
                &description,
                impact,
                game,
            )
            .await
            .unwrap_or_else(|e| format!("Council evaluation error: {}", e));

        self.apply_active_effects(game);

        Some(result)
    }

    fn choose_wise_proposal(
        &self,
        game: &PowrushGame,
    ) -> (CouncilFocus, String, String, WorldImpactType) {
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
                WorldImpactType::CulturalFestival,
            )
        } else if game.current_cycle % 7 == 0 {
            (
                CouncilFocus::HarmonyPreservation,
                "Great Harmony Festival".to_string(),
                "Strengthen bonds between all factions through a world-wide festival.".to_string(),
                WorldImpactType::FactionTreatySigned,
            )
        } else if collective_joy > 88.0 {
            (
                CouncilFocus::EternalCompassion,
                "Great Mercy Bloom".to_string(),
                "Collective joy is exceptionally high. Amplify mercy across the lattice.".to_string(),
                WorldImpactType::AllianceFormed,
            )
        } else {
            (
                CouncilFocus::AbundanceCreation,
                "Regional Resource Bloom".to_string(),
                "Create a large-scale abundance surge in multiple regions.".to_string(),
                WorldImpactType::FactionAIStrategies,
            )
        }
    }

    fn apply_active_effects(&mut self, game: &mut PowrushGame) {
        for change in self.governance_engine.active_changes.values() {
            match change.impact_type {
                WorldImpactType::CulturalFestival | WorldImpactType::AllianceFormed => {
                    for player in &mut game.players {
                        player.needs.joy =
                            (player.needs.joy + change.joy_boost * 0.4).min(100.0);
                    }
                }
                WorldImpactType::FactionAIStrategies => {
                    for resource in &mut game.resources {
                        resource.mercy_multiplier *= 1.12;
                    }
                }
                _ => {}
            }
        }

        self.governance_engine
            .cleanup_expired_effects(game.current_cycle);
    }

    pub fn get_status(&self) -> String {
        format!(
            "PATSAGi Governance v14.15.0 | cycles_since={} | frequency={} | interventions={} | active_changes={}",
            self.cycles_since_last_governance,
            self.governance_frequency,
            self.interventions,
            self.governance_engine.active_changes.len()
        )
    }
}

impl Default for SimulationIntegration {
    fn default() -> Self {
        Self::new()
    }
}

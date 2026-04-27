//! # Advanced Simulation Engine (v0.1.0)
//!
//! The multi-player, mercy-gated simulation heart of Powrush.
//! Handles faction diplomacy, collective resource flows, world events,
//! and real-time evaluation against the 7 Living Mercy Gates.
//!
//! This engine powers both Single-Player and future MMO modes.

use crate::game::PowrushGame;
use crate::player::Player;
use crate::faction::Faction;
use crate::mercy::{MercyGateStatus, evaluate_all_gates};
use crate::resources::ResourceType;
use rand::Rng;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldEvent {
    pub id: u64,
    pub description: String,
    pub mercy_impact: f64,
    pub affected_factions: Vec<Faction>,
    pub timestamp: DateTime<Utc>,
    pub joy_boost: f32,
}

pub struct SimulationEngine {
    pub game: PowrushGame,
    pub world_events: Vec<WorldEvent>,
    pub diplomacy_matrix: std::collections::HashMap<(Faction, Faction), f64>,
    pub collective_joy: f32,
}

impl SimulationEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            game: PowrushGame::new(),
            world_events: Vec::new(),
            diplomacy_matrix: std::collections::HashMap::new(),
            collective_joy: 72.0,
        };
        engine.initialize_diplomacy_matrix();
        engine
    }

    fn initialize_diplomacy_matrix(&mut self) {
        let factions = Faction::all_factions();
        for &f1 in &factions {
            for &f2 in &factions {
                if f1 != f2 {
                    let bonus = f1.get_diplomacy_bonus(f2);
                    self.diplomacy_matrix.insert((f1, f2), bonus);
                }
            }
        }
    }

    /// Run one full multi-player mercy-gated simulation cycle.
    pub async fn run_multi_player_cycle(&mut self) -> Result<String, String> {
        // 1. Run base game cycle (resources + mercy)
        let base_result = self.game.run_simulation_cycle().await?;

        // 2. Update collective joy (5-Gene Joy Tetrad influence)
        self.update_collective_joy();

        // 3. Process faction diplomacy & resource sharing
        self.process_faction_diplomacy().await;

        // 4. Generate dynamic world events
        self.generate_world_events().await;

        // 5. Apply mercy evaluation to every player
        for player in &mut self.game.players {
            let status = evaluate_all_gates(
                "participation in collective mercy cycle",
                "multi-player simulation",
                4.75,
                0.93,
            ).await?;

            player.apply_mercy_result(status);

            // Bonus from collective joy
            if self.collective_joy > 85.0 {
                player.needs.joy = (player.needs.joy + 3.0).min(100.0);
            }
        }

        Ok(format!(
            "Multi-player cycle {} complete.\n{}\nCollective Joy: {:.1} | World Events: {}",
            self.game.current_cycle,
            base_result,
            self.collective_joy,
            self.world_events.len()
        ))
    }

    fn update_collective_joy(&mut self) {
        let avg_happiness: f32 = if !self.game.players.is_empty() {
            self.game.players.iter().map(|p| p.happiness).sum::<f32>() / self.game.players.len() as f32
        } else {
            75.0
        };

        self.collective_joy = (self.collective_joy * 0.92 + avg_happiness * 0.08).clamp(40.0, 100.0);
    }

    async fn process_faction_diplomacy(&mut self) {
        // Future: real resource sharing, alliance bonuses, joint projects
        // For now: gentle positive drift when mercy gates are honored
        for ((f1, f2), bonus) in self.diplomacy_matrix.iter_mut() {
            if *bonus < 1.8 {
                *bonus *= 1.002; // Slow, mercy-driven improvement
            }
        }
    }

    async fn generate_world_events(&mut self) {
        let mut rng = rand::thread_rng();

        if rng.gen_bool(0.28) {
            let event = WorldEvent {
                id: rand::random(),
                description: "A massive Ambrosian Nectar bloom has occurred! Joy surges across the world.".to_string(),
                mercy_impact: 0.22,
                affected_factions: vec![Faction::Ambrosians, Faction::EternalCompassion],
                timestamp: Utc::now(),
                joy_boost: 12.0,
            };
            self.world_events.push(event);
            self.collective_joy = (self.collective_joy + 8.0).min(100.0);
        }

        if rng.gen_bool(0.18) {
            let event = WorldEvent {
                id: rand::random(),
                description: "The Harmonists and Mercy Weavers have completed a joint mercy project.".to_string(),
                mercy_impact: 0.18,
                affected_factions: vec![Faction::Harmonists, Faction::MercyWeavers],
                timestamp: Utc::now(),
                joy_boost: 7.0,
            };
            self.world_events.push(event);
        }
    }

    /// Get current world state summary
    pub fn get_world_summary(&self) -> String {
        format!(
            "Cycle: {} | Players: {} | Collective Joy: {:.1} | Events: {} | Abundance: {}",
            self.game.current_cycle,
            self.game.players.len(),
            self.collective_joy,
            self.world_events.len(),
            self.game.get_abundance_summary()
        )
    }
}

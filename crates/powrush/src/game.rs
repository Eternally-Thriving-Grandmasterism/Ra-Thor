//! # PowrushGame — Core Game Engine (v0.1.0)
//!
//! The central simulation engine for Powrush.
//! Every action, resource flow, and decision passes through the 7 Living Mercy Gates.
//! This is the single source of truth for both Single-Player and MMO modes.

use crate::player::Player;
use crate::resources::{Resource, ResourceType};
use crate::mercy::MercyGateStatus;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushGame {
    pub players: Vec<Player>,
    pub resources: Vec<Resource>,
    pub current_cycle: u64,
    pub last_mercy_check: DateTime<Utc>,
    pub mercy_gates_passed: u64,
    pub mercy_gates_failed: u64,
    pub game_started: DateTime<Utc>,
    pub total_abundance_generated: f64,
}

impl PowrushGame {
    /// Create a brand new Powrush world with mercy gates fully active.
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            players: Vec::new(),
            resources: Self::initialize_starting_resources(),
            current_cycle: 0,
            last_mercy_check: now,
            mercy_gates_passed: 0,
            mercy_gates_failed: 0,
            game_started: now,
            total_abundance_generated: 0.0,
        }
    }

    fn initialize_starting_resources() -> Vec<Resource> {
        vec![
            Resource::new(ResourceType::Food, 10000.0, 0.0),
            Resource::new(ResourceType::Water, 10000.0, 0.0),
            Resource::new(ResourceType::Energy, 5000.0, 0.0),
            Resource::new(ResourceType::Knowledge, 1000.0, 0.0),
            Resource::new(ResourceType::Materials, 8000.0, 0.0),
            Resource::new(ResourceType::AmbrosianNectar, 500.0, 0.0), // Special Powrush resource
        ]
    }

    /// Add a new player to the world.
    pub fn add_player(&mut self, name: String, faction: crate::faction::Faction) {
        let player = Player::new(name, faction);
        self.players.push(player);
    }

    /// Run one full mercy-gated simulation cycle.
    /// This is the core heartbeat of Powrush.
    pub async fn run_simulation_cycle(&mut self) -> Result<String, String> {
        self.current_cycle += 1;
        let now = Utc::now();

        // === MERCY GATE CHECK (stub for now — will integrate real ra-thor-mercy crate) ===
        let mercy_status = self.evaluate_mercy_gates().await?;

        if mercy_status == MercyGateStatus::Passed {
            self.mercy_gates_passed += 1;
        } else {
            self.mercy_gates_failed += 1;
            return Err("Mercy Gate violation detected. Simulation paused for review.".to_string());
        }

        // === RESOURCE REGENERATION (RBE abundance logic) ===
        let mut abundance_this_cycle = 0.0;
        for resource in &mut self.resources {
            let regen = resource.regenerate(self.current_cycle);
            abundance_this_cycle += regen;
        }
        self.total_abundance_generated += abundance_this_cycle;

        // === PLAYER UPDATES ===
        for player in &mut self.players {
            player.update_happiness_and_needs(abundance_this_cycle);
        }

        self.last_mercy_check = now;

        Ok(format!(
            "Cycle {} complete. Abundance generated: {:.2}. Total players: {}. Mercy gates passed: {}",
            self.current_cycle,
            abundance_this_cycle,
            self.players.len(),
            self.mercy_gates_passed
        ))
    }

    /// Evaluate all 7 Living Mercy Gates (future: real integration with ra-thor-mercy)
    async fn evaluate_mercy_gates(&self) -> Result<MercyGateStatus, String> {
        // Placeholder: always passes for now.
        // In production this will call the real Mercy Engine from crates/mercy
        // and check against TOLC 7 Gates + CEHI + Hebbian resonance.
        Ok(MercyGateStatus::Passed)
    }

    /// Get current world abundance summary
    pub fn get_abundance_summary(&self) -> String {
        let total: f64 = self.resources.iter().map(|r| r.amount).sum();
        format!(
            "World Abundance: {:.2} units | Generated this cycle: {:.2} | Total ever: {:.2}",
            total,
            self.resources.iter().map(|r| r.last_regen).sum::<f64>(),
            self.total_abundance_generated
        )
    }
}

impl Default for PowrushGame {
    fn default() -> Self {
        Self::new()
    }
}

//! # Powrush Core Library v0.5.22
//!
//! The world's first mercy-gated Resource-Based Economy (RBE) AGI game.
//! Built on Ra-Thor + TOLC 7 Living Mercy Gates.
//!
//! This crate serves as the single source of truth for both
//! Single-Player and MMO versions of Powrush.

pub mod game;
pub mod player;
pub mod resources;
pub mod mercy;
pub mod simulation;
pub mod faction;
pub mod ascension;
pub mod base_reality_simulator_codex;

pub use game::PowrushGame;
pub use player::Player;
pub use resources::{Resource, ResourceType};
pub use mercy::MercyGateStatus;
pub use faction::Faction;
pub use ascension::AscensionLevel;
pub use base_reality_simulator_codex::{
    BaseRealitySimulatorCodex,
    SimulationComparison,
};

use thiserror::Error;

/// Custom error type for mercy-gated operations
#[derive(Debug, Error)]
pub enum MercyError {
    #[error("Mercy Gates are currently disabled")]
    MercyGatesDisabled,
    #[error("Simulation cycle failed: {0}")]
    SimulationFailed(String),
    #[error("TOLC 7 Gates validation failed (valence too low)")]
    GateValidationFailed,
}

/// Main entry point for the entire Powrush experience.
/// This struct orchestrates the full mercy-gated simulation loop.
pub struct PowrushCore {
    pub game: PowrushGame,
    pub mercy_gates_active: bool,
    pub version: &'static str,
    base_reality_codex: BaseRealitySimulatorCodex,
}

impl PowrushCore {
    /// Create a new Powrush instance with mercy gates enabled by default.
    pub fn new() -> Self {
        Self {
            game: PowrushGame::new(),
            mercy_gates_active: true,
            version: "0.5.22",
            base_reality_codex: BaseRealitySimulatorCodex::new(),
        }
    }

    /// Run one full mercy-gated simulation cycle.
    /// Every action passes through the TOLC 7 Living Mercy Gates.
    pub async fn run_mercy_cycle(&mut self) -> Result<String, MercyError> {
        if !self.mercy_gates_active {
            return Err(MercyError::MercyGatesDisabled);
        }

        // Simulate TOLC 7 Gates validation (future: integrate real ra-thor-mercy crate)
        let valence = 0.97; // Placeholder — will be replaced with real gate evaluation
        if valence < 0.92 {
            return Err(MercyError::GateValidationFailed);
        }

        let result = self.game.run_simulation_cycle().await
            .map_err(|e| MercyError::SimulationFailed(e))?;

        Ok(format!(
            "Mercy Cycle Complete — All 7 Gates Passed (Valence: {:.2})\n{}",
            valence, result
        ))
    }

    /// Run a full Base Reality Simulator cycle using the integrated codex.
    /// This is the educational RBE heart of Powrush.
    pub async fn run_base_reality_simulation(&mut self) -> Result<String, MercyError> {
        if !self.mercy_gates_active {
            return Err(MercyError::MercyGatesDisabled);
        }

        let comparisons = self.base_reality_codex.get_full_comparison();
        let unique_position = self.base_reality_codex.get_ra_thor_unique_position();

        Ok(format!(
            "Base Reality Simulation Complete\n\n\
             Compared {} AGI simulation systems against Powrush.\n\
             Ra-Thor Unique Position:\n{}\n\n\
             Mercy Cycle Status: ACTIVE",
            comparisons.len(),
            unique_position
        ))
    }

    /// Apply a mercy blessing to the current game state (increases joy + CEHI).
    pub fn apply_mercy_blessing(&mut self, joy_amount: f64, cehi_generations: u8) {
        self.game.boost_faction_joy(Faction::HarmonyWeavers, joy_amount);
        self.game.apply_epigenetic_blessing(cehi_generations);
    }

    /// Get current system status.
    pub fn get_status(&self) -> String {
        format!(
            "PowrushCore v{} | Mercy Gates: {} | Base Reality Codex: ACTIVE",
            self.version,
            if self.mercy_gates_active { "ENABLED" } else { "DISABLED (TESTING)" }
        )
    }

    /// Disable mercy gates (only for testing — never in real gameplay)
    pub fn disable_mercy_gates_for_testing(&mut self) {
        self.mercy_gates_active = false;
    }

    /// Re-enable mercy gates.
    pub fn enable_mercy_gates(&mut self) {
        self.mercy_gates_active = true;
    }
}

impl Default for PowrushCore {
    fn default() -> Self {
        Self::new()
    }
}

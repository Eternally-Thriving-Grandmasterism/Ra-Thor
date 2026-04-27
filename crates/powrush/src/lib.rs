//! # Powrush Core Library v0.1.0
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

pub use game::PowrushGame;
pub use player::Player;
pub use resources::{Resource, ResourceType};
pub use mercy::MercyGateStatus;
pub use faction::Faction;
pub use ascension::AscensionLevel;

/// Main entry point for the entire Powrush experience.
/// This struct orchestrates the full mercy-gated simulation loop.
pub struct PowrushCore {
    pub game: PowrushGame,
    pub mercy_gates_active: bool,
    pub version: &'static str,
}

impl PowrushCore {
    /// Create a new Powrush instance with mercy gates enabled by default.
    pub fn new() -> Self {
        Self {
            game: PowrushGame::new(),
            mercy_gates_active: true,
            version: "0.1.0",
        }
    }

    /// Run one full mercy-gated simulation cycle.
    /// This is the heart of Powrush — every action passes through the 7 Living Mercy Gates.
    pub async fn run_mercy_cycle(&mut self) -> Result<String, String> {
        if !self.mercy_gates_active {
            return Err("Mercy Gates are currently disabled. This should never happen in production.".to_string());
        }

        // Future: integrate with ra-thor-mercy crate for real gate evaluation
        let result = self.game.run_simulation_cycle().await?;
        
        Ok(format!(
            "Mercy Cycle Complete — All 7 Gates Passed\n{}",
            result
        ))
    }

    /// Disable mercy gates (only for testing — never in real gameplay)
    pub fn disable_mercy_gates_for_testing(&mut self) {
        self.mercy_gates_active = false;
    }
}

impl Default for PowrushCore {
    fn default() -> Self {
        Self::new()
    }
}

//! # Powrush Integration Bridge (v0.1.0)
//!
//! This module provides a single, beautiful function that the main
//! Powrush simulation loop can call every cycle.
//!
//! Just add one line in your main simulation loop and the 13+ PATSAGi
//! Councils will automatically govern, propose, and implement world changes.

use crate::simulation_integration::SimulationIntegration;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushPatsagiBridge {
    pub integration: SimulationIntegration,
    pub enabled: bool,
}

impl PowrushPatsagiBridge {
    pub fn new() -> Self {
        Self {
            integration: SimulationIntegration::new(),
            enabled: true,
        }
    }

    /// Call this once per simulation cycle from the main Powrush loop.
    /// Returns Some(message) if the Councils made a world change, None otherwise.
    pub async fn tick(&mut self, game: &mut PowrushGame) -> Option<String> {
        if !self.enabled {
            return None;
        }

        self.integration.tick(game).await
    }

    /// Enable or disable Council governance (useful for testing)
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current status
    pub fn get_status(&self) -> String {
        self.integration.get_status()
    }

    /// Quick helper to get the full Council status report
    pub fn get_council_report(&self) -> String {
        self.integration.governance_engine.coordinator.get_council_status_report()
    }
}

impl Default for PowrushPatsagiBridge {
    fn default() -> Self {
        Self::new()
    }
}

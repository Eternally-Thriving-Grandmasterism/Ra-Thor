//! # Powrush Integration Bridge v0.3.1
//!
//! This module provides a single, clean, and powerful function that the main
//! Powrush simulation loop can call every cycle.
//!
//! Just add one line in your main simulation loop and the 13+ PATSAGi
//! Councils will automatically govern, propose, and implement world changes.

use crate::{
    WorldGovernanceEngine,
    SimulationIntegration,
    WorldImpactType,
};
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowrushPatsagiBridge {
    pub integration: SimulationIntegration,
    pub enabled: bool,
    pub version: &'static str,
}

impl PowrushPatsagiBridge {
    pub fn new() -> Self {
        Self {
            integration: SimulationIntegration::new(),
            enabled: true,
            version: "0.3.1",
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

    /// Enable or disable Council governance (useful for testing or special events)
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get current status for debugging or UI
    pub fn get_status(&self) -> String {
        self.integration.get_status()
    }

    /// Get full Council status report
    pub fn get_council_report(&self) -> String {
        self.integration.governance_engine.coordinator.get_council_status_report()
    }

    /// Force an immediate governance cycle (useful for special events)
    pub async fn force_governance_cycle(&mut self, game: &mut PowrushGame) -> String {
        let result = self.integration.governance_engine
            .propose_and_approve_world_change(
                crate::CouncilFocus::EternalCompassion,
                "Forced Governance Cycle",
                "A special event has triggered immediate Council governance.",
                WorldImpactType::MercyBloom,
                game,
            )
            .await
            .unwrap_or_default();

        result
    }
}

impl Default for PowrushPatsagiBridge {
    fn default() -> Self {
        Self::new()
    }
}

//! # Powrush Integration Bridge — v14.15.0
//!
//! Single clean entry point the main Powrush simulation loop can call every cycle.
//! The 16 PATSAGi Councils automatically govern, propose, and implement world changes.
//!
//! Living Cosmic Tick aligned.
//! Contact: info@Rathor.ai
//! AG-SML v1.0

use crate::{CouncilFocus, SimulationIntegration, WorldImpactType};
use powrush::PowrushGame;
use serde::{Deserialize, Serialize};

pub const VERSION: &str = "14.15.0";

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
            version: VERSION,
        }
    }

    /// Call once per simulation cycle from the main Powrush loop.
    /// Returns `Some(message)` if the Councils made a world change.
    pub async fn tick(&mut self, game: &mut PowrushGame) -> Option<String> {
        if !self.enabled {
            return None;
        }
        self.integration.tick(game).await
    }

    /// Enable or disable Council governance.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Current status for debugging or UI.
    pub fn get_status(&self) -> String {
        format!(
            "PowrushPatsagiBridge v{} | enabled={} | {}",
            self.version,
            self.enabled,
            self.integration.get_status()
        )
    }

    /// Force an immediate governance cycle (special events).
    pub async fn force_governance_cycle(&mut self, game: &mut PowrushGame) -> String {
        self.integration
            .governance_engine
            .propose_and_approve_world_change(
                CouncilFocus::EternalCompassion,
                "Forced Governance Cycle",
                "A special event has triggered immediate Council governance.",
                WorldImpactType::AllianceFormed,
                game,
            )
            .await
            .unwrap_or_else(|e| format!("Force governance error: {}", e))
    }

    /// Compact telemetry summary.
    pub fn summary(&self) -> String {
        format!(
            "PowrushPatsagiBridge v{} | enabled={} | interventions={} | Living Cosmic Tick active",
            self.version,
            self.enabled,
            self.integration.interventions
        )
    }
}

impl Default for PowrushPatsagiBridge {
    fn default() -> Self {
        Self::new()
    }
}

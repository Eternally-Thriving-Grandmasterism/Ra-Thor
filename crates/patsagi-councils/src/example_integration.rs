//! # Example Integration — v14.15.0
//!
//! How to wire the 16 PATSAGi Councils into the Powrush-MMO simulation loop.
//!
//! Copy this pattern into `crates/powrush/src/simulation.rs` (or any host loop)
//! to let the Councils govern the world automatically.
//!
//! Contact: info@Rathor.ai

use patsagi_councils::PowrushPatsagiBridge;
use powrush::PowrushGame;

pub struct EnhancedPowrushSimulation {
    pub game: PowrushGame,
    pub council_bridge: PowrushPatsagiBridge,
}

impl EnhancedPowrushSimulation {
    pub fn new() -> Self {
        Self {
            game: PowrushGame::new(),
            council_bridge: PowrushPatsagiBridge::new(),
        }
    }

    /// Call every simulation tick.
    pub async fn run_enhanced_cycle(&mut self) -> Result<String, String> {
        // 1. Normal Powrush simulation cycle
        let normal_result = self.game.run_simulation_cycle().await?;

        // 2. PATSAGi Councils governance tick
        let council_message = self.council_bridge.tick(&mut self.game).await;

        // 3. Combine
        let mut full_result = normal_result;
        if let Some(council_msg) = council_message {
            full_result.push_str("\n\n");
            full_result.push_str(&council_msg);
        }

        Ok(full_result)
    }

    /// Council status for UI or debugging.
    pub fn get_council_status(&self) -> String {
        self.council_bridge.get_status()
    }

    pub fn summary(&self) -> String {
        format!(
            "EnhancedPowrushSimulation v14.15.0 | {}",
            self.council_bridge.summary()
        )
    }
}

impl Default for EnhancedPowrushSimulation {
    fn default() -> Self {
        Self::new()
    }
}

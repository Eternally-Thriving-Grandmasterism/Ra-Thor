//! # Example Integration — How to Use the 13+ PATSAGi Councils in Powrush-MMO
//!
//! This file shows the exact code you can copy into
//! `crates/powrush/src/simulation.rs` (or any simulation loop)
//! to make the Councils govern the world automatically.

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

    /// Call this every simulation tick (exactly like the normal run_simulation_cycle)
    pub async fn run_enhanced_cycle(&mut self) -> Result<String, String> {
        // 1. Run normal Powrush simulation cycle
        let normal_result = self.game.run_simulation_cycle().await?;

        // 2. Let the 13+ PATSAGi Councils govern and possibly make world changes
        let council_message = self.council_bridge.tick(&mut self.game).await;

        // 3. Combine results beautifully
        let mut full_result = normal_result;

        if let Some(council_msg) = council_message {
            full_result.push_str("\n\n");
            full_result.push_str(&council_msg);
        }

        Ok(full_result)
    }

    /// Get current Council status for UI or debugging
    pub fn get_council_status(&self) -> String {
        self.council_bridge.get_status()
    }
}

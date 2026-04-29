//! # Mercy Engine Adapter v0.5.9
//!
//! Unifies the Advanced MercyEngine (current rich logic used in PatsagiCouncilCoordinator)
//! with the Modular MercyEngine (live monorepo self-evolving, music-mercy, TOLC-native version).
//! Controlled by the `modular-mercy` feature flag.

use mercy::MercyEngine;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyEngineVariant {
    /// Current rich logic used in PatsagiCouncilCoordinator and voting system
    Advanced,
    /// Live monorepo modular mercy crate (self-evolving, music-mercy, TOLC-native)
    Modular,
}

pub struct MercyEngineAdapter {
    pub variant: MercyEngineVariant,
    advanced_engine: MercyEngine,
}

impl MercyEngineAdapter {
    pub fn new(variant: MercyEngineVariant) -> Self {
        Self {
            variant,
            advanced_engine: MercyEngine::new(),
        }
    }

    pub async fn evaluate_action(
        &self,
        action: &str,
        context: &str,
        cehi: f64,
        mercy_valence: f64,
    ) -> Result<mercy::MercyGateStatus, String> {
        match self.variant {
            MercyEngineVariant::Advanced => {
                self.advanced_engine.evaluate_action(action, context, cehi, mercy_valence).await
            }
            MercyEngineVariant::Modular => {
                // Future-proof: when mercy crate fully exposes modular API, delegate here.
                // For now falls back to Advanced (safe and compatible).
                self.advanced_engine.evaluate_action(action, context, cehi, mercy_valence).await
            }
        }
    }

    pub fn get_variant(&self) -> MercyEngineVariant {
        self.variant
    }
}

impl Default for MercyEngineAdapter {
    fn default() -> Self {
        Self::new(MercyEngineVariant::Advanced)
    }
}

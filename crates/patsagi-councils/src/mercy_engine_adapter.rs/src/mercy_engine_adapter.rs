//! # Mercy Engine Adapter v0.5.8
//!
//! Unifies the Advanced MercyEngine (current rich logic) with the Modular MercyEngine (live monorepo).
//! Controlled by the `modular-mercy` feature flag.

use mercy::MercyEngine;
use powrush::PowrushGame;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MercyEngineVariant {
    Advanced,   // Current rich logic used in PatsagiCouncilCoordinator
    Modular,    // Live monorepo modular mercy crate (self-evolving, music-mercy, TOLC-native)
}

pub struct MercyEngineAdapter {
    pub variant: MercyEngineVariant,
    advanced_engine: MercyEngine,
    // modular_engine: Option<mercy::modular::ModularMercyEngine>, // ready for future when modular is fully exposed
}

impl MercyEngineAdapter {
    pub fn new(variant: MercyEngineVariant) -> Self {
        Self {
            variant,
            advanced_engine: MercyEngine::new(),
            // modular_engine: if variant == MercyEngineVariant::Modular { Some(mercy::modular::ModularMercyEngine::new()) } else { None },
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
                // When modular feature is enabled and mercy crate exposes it, delegate here
                // For now falls back to advanced (future-proof)
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

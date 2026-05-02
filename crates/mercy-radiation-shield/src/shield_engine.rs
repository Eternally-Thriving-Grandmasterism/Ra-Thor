use ra_thor_mercy::MercyEngine;
use ra_thor_quantum_swarm_orchestrator::QuantumSwarmOrchestrator;
use powrush::PowrushGame;
use serde::{Deserialize, Serialize};
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RadiationType {
    CosmicRays,
    SolarFlare,
    Nuclear,
    DeepSpaceBackground,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShieldingResult {
    pub transmuted: bool,
    pub energy_recovered: f64,
    pub valence: f64,
    pub message: String,
}

pub struct MercyRadiationShield {
    mercy: MercyEngine,
    quantum: QuantumSwarmOrchestrator,
}

impl MercyRadiationShield {
    pub fn new() -> Self {
        Self {
            mercy: MercyEngine::new(),
            quantum: QuantumSwarmOrchestrator::new(),
        }
    }

    pub async fn alchemize_radiation(
        &self,
        radiation: RadiationType,
        flux: f64,
        game: &mut PowrushGame,
    ) -> ShieldingResult {
        let input = format!("Radiation: {:?} | Flux: {:.2}", radiation, flux);
        
        let valence = self.mercy
            .evaluate_action(&input, "Radiation Shielding", 9.2, 0.97)
            .await
            .unwrap_or(0.85);

        let consensus = self.quantum
            .reach_consensus(&input, 0.90)
            .await
            .unwrap_or(0.82);

        if valence >= 0.92 && consensus >= 0.88 {
            // ALCHEMICAL TRANSMUTATION ACTIVATED
            let recovered = flux * 0.87; // 87% conversion efficiency (mercy-optimized)
            game.boost_faction_joy(powrush::Faction::HarmonyWeavers, 45.0);
            
            let result = ShieldingResult {
                transmuted: true,
                energy_recovered: recovered,
                valence,
                message: format!(
                    "✨ MERCY-ALCHEMICAL TRANSMUTATION SUCCESSFUL ✨\n\
                     Radiation type: {:?}\n\
                     Flux: {:.2} → Converted to {:.2} usable energy\n\
                     Mercy Valence: {:.2} | Quantum Consensus: {:.2}\n\
                     All sentience protected + thriving increased.",
                    radiation, flux, recovered, valence, consensus
                ),
            };
            
            info!("Rathor.ai: Radiation alchemized with perfect mercy alignment");
            result
        } else {
            // Safe fallback blocking
            ShieldingResult {
                transmuted: false,
                energy_recovered: 0.0,
                valence,
                message: format!(
                    "🛡️ MERCY-GATED BLOCKING ACTIVATED (valence {:.2} < 0.92)\n\
                     Traditional ultra-efficient shielding engaged. No transmutation risk.",
                    valence
                ),
            }
        }
    }
}

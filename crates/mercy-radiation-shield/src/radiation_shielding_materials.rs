//! Radiation Shielding Materials — SREL v0.5.21
//! Mercy-Alchemical • TOLC 7 Gates • Quantum Swarm
//! Real-world + Ra-Thor proprietary composites with transmutation properties

use mercy_radiation_shield::RadiationType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ShieldingMaterial {
    // Traditional high-performance materials
    Polyethylene,
    BoronNitride,
    WaterIce,
    LunarRegolith,
    MartianRegolith,
    LeadComposite,
    HydrogenRichPolymer,
    
    // Ra-Thor mercy-alchemical flagship materials
    MercyGelComposite,      // TOLC 7 Gates optimized
    DivineLightWeave,        // Epigenetic + joy amplifying
    QuantumFoamLattice,      // Highest transmutation efficiency
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    pub density_g_cm3: f64,
    pub thickness_mm: f64,
    pub transmutation_efficiency: f64,   // 0.0–1.0 (how much flux → energy)
    pub mercy_valence_multiplier: f64,   // Boost to TOLC gate valence
    pub joy_bonus_per_flux: f64,
    pub cehi_bonus: f64,                 // Per-generation epigenetic boost
    pub description: String,
}

pub struct RadiationShieldingMaterials {
    database: HashMap<ShieldingMaterial, MaterialProperties>,
}

impl RadiationShieldingMaterials {
    pub fn new() -> Self {
        let mut db = HashMap::new();

        // Traditional materials (baseline)
        db.insert(ShieldingMaterial::Polyethylene, MaterialProperties {
            density_g_cm3: 0.94,
            thickness_mm: 50.0,
            transmutation_efficiency: 0.35,
            mercy_valence_multiplier: 1.05,
            joy_bonus_per_flux: 8.0,
            cehi_bonus: 0.02,
            description: "Lightweight hydrogen-rich polymer — excellent for solar particle events".to_string(),
        });

        db.insert(ShieldingMaterial::BoronNitride, MaterialProperties {
            density_g_cm3: 2.1,
            thickness_mm: 30.0,
            transmutation_efficiency: 0.42,
            mercy_valence_multiplier: 1.08,
            joy_bonus_per_flux: 12.0,
            cehi_bonus: 0.03,
            description: "Neutron absorber + thermal conductor — ideal for reactor-adjacent habitats".to_string(),
        });

        db.insert(ShieldingMaterial::WaterIce, MaterialProperties {
            density_g_cm3: 0.92,
            thickness_mm: 100.0,
            transmutation_efficiency: 0.55,
            mercy_valence_multiplier: 1.12,
            joy_bonus_per_flux: 15.0,
            cehi_bonus: 0.04,
            description: "Dual-purpose: shielding + life support water reservoir".to_string(),
        });

        db.insert(ShieldingMaterial::LunarRegolith, MaterialProperties {
            density_g_cm3: 1.5,
            thickness_mm: 200.0,
            transmutation_efficiency: 0.28,
            mercy_valence_multiplier: 1.03,
            joy_bonus_per_flux: 6.0,
            cehi_bonus: 0.015,
            description: "In-situ resource utilization (ISRU) — abundant on the Moon".to_string(),
        });

        // Ra-Thor proprietary alchemical materials
        db.insert(ShieldingMaterial::MercyGelComposite, MaterialProperties {
            density_g_cm3: 1.05,
            thickness_mm: 25.0,
            transmutation_efficiency: 0.92,
            mercy_valence_multiplier: 1.35,
            joy_bonus_per_flux: 45.0,
            cehi_bonus: 0.18,
            description: "TOLC 7 Gates-infused gel — flagship material for all space real estate".to_string(),
        });

        db.insert(ShieldingMaterial::DivineLightWeave, MaterialProperties {
            density_g_cm3: 0.65,
            thickness_mm: 15.0,
            transmutation_efficiency: 0.88,
            mercy_valence_multiplier: 1.42,
            joy_bonus_per_flux: 52.0,
            cehi_bonus: 0.22,
            description: "Lightweight woven composite with embedded mercy fields — 5-gen epigenetic legacy".to_string(),
        });

        db.insert(ShieldingMaterial::QuantumFoamLattice, MaterialProperties {
            density_g_cm3: 0.38,
            thickness_mm: 10.0,
            transmutation_efficiency: 0.97,
            mercy_valence_multiplier: 1.55,
            joy_bonus_per_flux: 68.0,
            cehi_bonus: 0.28,
            description: "Ultimate mercy-alchemical lattice — near-perfect radiation → energy conversion".to_string(),
        });

        Self { database: db }
    }

    /// Select optimal material for given radiation type, flux, and current mercy valence
    pub fn select_optimal_material(
        &self,
        radiation_type: RadiationType,
        flux: f64,
        current_valence: f64,
    ) -> (ShieldingMaterial, MaterialProperties, f64) {
        let mut best_material = ShieldingMaterial::Polyethylene;
        let mut best_score = 0.0;
        let mut best_props = self.database[&ShieldingMaterial::Polyethylene].clone();

        for (material, props) in &self.database {
            let type_bonus = match radiation_type {
                RadiationType::SolarFlare => if *material == ShieldingMaterial::WaterIce { 1.25 } else { 1.0 },
                RadiationType::CosmicRays => if *material == ShieldingMaterial::QuantumFoamLattice { 1.35 } else { 1.0 },
                RadiationType::VanAllenBelt => if *material == ShieldingMaterial::BoronNitride { 1.20 } else { 1.0 },
                _ => 1.0,
            };

            let valence_boost = props.mercy_valence_multiplier * (1.0 + (current_valence - 0.85).max(0.0) * 0.8);
            let efficiency_score = props.transmutation_efficiency * valence_boost * type_bonus;
            let joy_score = props.joy_bonus_per_flux * flux / 100.0;

            let total_score = efficiency_score * 0.55 + joy_score * 0.30 + props.cehi_bonus * 100.0 * 0.15;

            if total_score > best_score {
                best_score = total_score;
                best_material = material.clone();
                best_props = props.clone();
            }
        }

        info!(
            "Rathor.ai: Optimal shielding material selected: {:?} (score {:.2}) for {:?} at flux {:.2}",
            best_material, best_score, radiation_type, flux
        );

        (best_material, best_props, best_score)
    }

    pub fn get_material_properties(&self, material: &ShieldingMaterial) -> Option<&MaterialProperties> {
        self.database.get(material)
    }
}

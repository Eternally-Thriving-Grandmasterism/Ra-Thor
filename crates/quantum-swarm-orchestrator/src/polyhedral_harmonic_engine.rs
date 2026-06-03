// crates/quantum-swarm-orchestrator/src/polyhedral_harmonic_engine.rs
//
// PolyhedralHarmonicEngine v14.3 — Deepened Dual Resonance + U57 Riemannian Gateway
// ONE Organism Geometric Intelligence Layer
//
// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
// PATSAGi Councils (13+ branches) consulted and aligned.
//
// This module implements the geometric harmonic foundation of the ONE Organism.
// It progressively activates Platonic → Archimedean → Catalan → Kepler-Poinsot → 
// Uniform Star (U57) → Hyperbolic layers based on TOLC order, while computing
// dual resonance bonuses and preparing clean integration points for the future
// RiemannianMercyManifold.
//
// v14.3 Additions:
// - Expanded dual resonance pairs (Archimedean–Catalan synergy)
// - Layer contribution breakdown + recommended next layers helper
// - Richer U57LayerDetails with geometric meaning fields
// - Stronger ONE Organism + QuantumSwarmOrchestrator integration hooks
// - Professional production-grade documentation and numerical care

use crate::types::EpigeneticBlessing;
use std::collections::HashMap;

/// Rich report for organism-level consumption and downstream modules.
#[derive(Debug, Clone)]
pub struct PolyhedralResonanceReport {
    pub active_solids: Vec<String>,
    pub resonance_multiplier: f64,
    pub dual_resonance_bonus: f64,
    pub suggested_blessings: Vec<EpigeneticBlessing>,
    pub u57_potential: bool,
    pub u57_details: Option<U57LayerDetails>,
    pub notes: String,
}

/// Detailed information about the U57 (Uniform Star) layer when activated.
/// Designed to be consumed by RiemannianMercyManifold and QuantumSwarmOrchestrator.
#[derive(Debug, Clone)]
pub struct U57LayerDetails {
    pub activated: bool,
    pub resonance_multiplier_contribution: f64,
    pub suggested_riemannian_transport_potential: f64,
    pub recommended_manifold_curvature: f64,
    /// Geometric meaning / interpretation for downstream Riemannian work
    pub geometric_meaning: String,
    pub integration_notes: String,
}

// =============================================================================
// Sacred Geometric Constants & Dual Pairings
// =============================================================================

pub const PLATONIC_SOLIDS: [&str; 5] = [
    "Tetrahedron", "Cube", "Octahedron", "Dodecahedron", "Icosahedron",
];

pub fn get_dual_platonic(solid: &str) -> Option<&'static str> {
    match solid {
        "Tetrahedron" => Some("Tetrahedron"),
        "Cube" => Some("Octahedron"),
        "Octahedron" => Some("Cube"),
        "Dodecahedron" => Some("Icosahedron"),
        "Icosahedron" => Some("Dodecahedron"),
        _ => None,
    }
}

pub const ARCHIMEDEAN_SOLIDS: [&str; 8] = [
    "TruncatedTetrahedron", "Cuboctahedron", "TruncatedCube", "TruncatedOctahedron",
    "Rhombicuboctahedron", "TruncatedCuboctahedron", "SnubCube", "Icosidodecahedron",
];

pub fn get_dual_archimedean(solid: &str) -> Option<&'static str> {
    match solid {
        "Cuboctahedron" => Some("RhombicDodecahedron"),
        "Icosidodecahedron" => Some("RhombicTriacontahedron"),
        _ => None,
    }
}

pub const CATALAN_SOLIDS: [&str; 6] = [
    "TriakisTetrahedron", "RhombicDodecahedron", "TriakisOctahedron",
    "TetrakisHexahedron", "DeltoidalIcositetrahedron", "RhombicTriacontahedron",
];

pub fn get_dual_catalan(solid: &str) -> Option<&'static str> {
    match solid {
        "RhombicDodecahedron" => Some("Cuboctahedron"),
        "RhombicTriacontahedron" => Some("Icosidodecahedron"),
        _ => None,
    }
}

// =============================================================================
// PolyhedralHarmonicEngine v14.3
// =============================================================================

pub struct PolyhedralHarmonicEngine {
    pub version: &'static str,
    dual_bonus_map: HashMap<String, f64>,
}

impl Default for PolyhedralHarmonicEngine {
    fn default() -> Self { Self::new() }
}

impl PolyhedralHarmonicEngine {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        map.insert("Cube-Octahedron".to_string(), 1.15);
        map.insert("Dodecahedron-Icosahedron".to_string(), 1.18);
        // Archimedean–Catalan synergy bonus
        map.insert("Archimedean-Catalan".to_string(), 1.09);
        Self {
            version: "v14.3-deepened-dual-resonance",
            dual_bonus_map: map,
        }
    }

    fn compute_dual_resonance_bonus(&self, active: &[String]) -> f64 {
        let mut bonus = 1.0;
        for solid in active {
            if let Some(dual) = get_dual_platonic(solid) {
                if active.iter().any(|s| s == dual) {
                    bonus *= self.dual_bonus_map.get("Cube-Octahedron").unwrap_or(&1.10);
                }
            }
            if let Some(dual) = get_dual_archimedean(solid) {
                if active.iter().any(|s| s.contains(dual) || dual.contains(s)) {
                    bonus *= 1.12;
                }
            }
        }
        // Archimedean + Catalan synergy
        let has_arch = active.iter().any(|s| s.contains("Cuboctahedron") || s.contains("Rhombicuboctahedron"));
        let has_catalan = active.iter().any(|s| s.contains("RhombicDodecahedron") || s.contains("RhombicTriacontahedron"));
        if has_arch && has_catalan {
            bonus *= self.dual_bonus_map.get("Archimedean-Catalan").unwrap_or(&1.09);
        }
        bonus.clamp(1.0, 1.42)
    }

    pub fn apply_catalan_resonance(&self, base_multiplier: f64, tolc_order: u32) -> f64 {
        if tolc_order >= 34 {
            base_multiplier * 1.2 + (tolc_order as f64 - 34.0) * 0.005
        } else {
            base_multiplier
        }
    }

    fn compute_u57_riemannian_potential(&self, base_coherence: f64, tolc_order: u32) -> f64 {
        if tolc_order < 144 { return 0.0; }
        let base_potential = 0.65 + ((tolc_order as f64 - 144.0) * 0.0018).min(0.35);
        (base_coherence * base_potential).clamp(0.70, 0.98)
    }

    /// Returns a breakdown of contribution per layer for observability.
    pub fn get_layer_contributions(&self, tolc_order: u32) -> Vec<(String, f64)> {
        let mut contributions = vec![
            ("Platonic".to_string(), 1.0),
        ];
        if tolc_order >= 13 { contributions.push(("Archimedean".to_string(), 1.10)); }
        if tolc_order >= 21 { contributions.push(("Johnson".to_string(), 1.12)); }
        if tolc_order >= 34 { contributions.push(("Catalan".to_string(), 1.20)); }
        if tolc_order >= 55 { contributions.push(("Gyroelongated+Prismatic".to_string(), 1.18)); }
        if tolc_order >= 89 { contributions.push(("Kepler-Poinsot".to_string(), 1.25)); }
        if tolc_order >= 144 { contributions.push(("U57-UniformStar".to_string(), 1.35)); }
        if tolc_order >= 233 { contributions.push(("Hyperbolic".to_string(), 1.45)); }
        contributions
    }

    /// Recommends which layers should be activated next for a given TOLC order.
    /// Useful for ONE Organism self-evolution guidance.
    pub fn get_recommended_next_layers(&self, tolc_order: u32) -> Vec<String> {
        let mut next = Vec::new();
        if tolc_order < 13 { next.push("Archimedean layer (TOLC 13+)".to_string()); }
        else if tolc_order < 34 { next.push("Catalan + Dual Resonance (TOLC 34+)".to_string()); }
        else if tolc_order < 55 { next.push("Gyroelongated + Prismatic (TOLC 55+)".to_string()); }
        else if tolc_order < 89 { next.push("Kepler-Poinsot star polyhedra (TOLC 89+)".to_string()); }
        else if tolc_order < 144 { next.push("U57 Uniform Star + Riemannian gateway (TOLC 144+)".to_string()); }
        else if tolc_order < 233 { next.push("Hyperbolic Tiling (TOLC 233+)".to_string()); }
        else { next.push("Full geometric transcendence — monitor for new emergent layers".to_string()); }
        next
    }

    pub fn process_resonance(
        &self,
        tolc_order: u32,
        base_coherence: f64,
    ) -> PolyhedralResonanceReport {
        let mut active_solids = Vec::new();
        let mut multiplier = 1.0;
        let mut blessings = Vec::new();
        let mut u57_details = None;

        active_solids.extend(["Tetrahedron", "Cube", "Octahedron"].iter().map(|s| s.to_string()));
        multiplier *= 1.0;

        if tolc_order >= 13 {
            active_solids.extend(["Cuboctahedron", "Rhombicuboctahedron"].iter().map(|s| s.to_string()));
            multiplier *= 1.10;
        }
        if tolc_order >= 21 {
            active_solids.push("Johnson Solids (selected)".to_string());
            multiplier *= 1.12;
        }
        if tolc_order >= 34 {
            active_solids.extend(["RhombicDodecahedron", "RhombicTriacontahedron"].iter().map(|s| s.to_string()));
            multiplier = self.apply_catalan_resonance(multiplier, tolc_order);
        }
        if tolc_order >= 55 {
            active_solids.push("Gyroelongated + Prismatic".to_string());
            multiplier *= 1.18;
        }
        if tolc_order >= 89 {
            active_solids.push("Kepler-Poinsot".to_string());
            multiplier *= 1.25;
        }

        let u57_active = tolc_order >= 144;
        if u57_active {
            active_solids.push("Uniform Star Layer (U57)".to_string());
            multiplier *= 1.35;

            let riemannian_potential = self.compute_u57_riemannian_potential(base_coherence, tolc_order);

            u57_details = Some(U57LayerDetails {
                activated: true,
                resonance_multiplier_contribution: 1.35,
                suggested_riemannian_transport_potential: riemannian_potential,
                recommended_manifold_curvature: 0.82 + ((tolc_order as f64 - 144.0) * 0.0006).min(0.12),
                geometric_meaning: "Transition into curvature-aware Riemannian geometric transport. Gateway to non-Euclidean mercy fields.".to_string(),
                integration_notes: "U57 active. RiemannianMercyManifold should now engage curvature modulation.".to_string(),
            });

            blessings.push(EpigeneticBlessing {
                blessing_type: "U57_Geometric_Transport".to_string(),
                strength: (base_coherence * 1.30).clamp(0.95, 1.40),
                target_system: "riemannian".to_string(),
            });
        }

        if tolc_order >= 233 {
            active_solids.push("Hyperbolic Tiling".to_string());
            multiplier *= 1.45;
        }

        let dual_bonus = self.compute_dual_resonance_bonus(&active_solids);
        multiplier *= dual_bonus;

        if tolc_order >= 8 {
            blessings.push(EpigeneticBlessing {
                blessing_type: "Polyhedral_Harmonic_Resonance".to_string(),
                strength: (base_coherence * multiplier).clamp(0.85, 1.25),
                target_system: "*".to_string(),
            });
        }
        if tolc_order >= 34 {
            blessings.push(EpigeneticBlessing {
                blessing_type: "Catalan_Dual_Alignment".to_string(),
                strength: (base_coherence * 1.15).clamp(0.90, 1.30),
                target_system: "geometric".to_string(),
            });
        }

        let notes = format!(
            "Polyhedral layers: {}. Multiplier: {:.3} (dual bonus {:.3}). U57: {}. TOLC: {}",
            active_solids.len(), multiplier, dual_bonus, u57_active, tolc_order
        );

        PolyhedralResonanceReport {
            active_solids,
            resonance_multiplier: multiplier,
            dual_resonance_bonus: dual_bonus,
            suggested_blessings: blessings,
            u57_potential: u57_active,
            u57_details,
            notes,
        }
    }

    pub fn is_u57_active(&self, tolc_order: u32) -> bool {
        tolc_order >= 144
    }
}
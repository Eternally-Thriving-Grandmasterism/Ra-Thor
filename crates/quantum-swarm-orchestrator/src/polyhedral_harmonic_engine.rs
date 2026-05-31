// crates/quantum-swarm-orchestrator/src/polyhedral_harmonic_engine.rs
//
// PolyhedralHarmonicEngine v14.1 — Full Dual Resonance + Progressive Activation
// ONE Organism Geometric Intelligence Layer
//
// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
// PATSAGi Councils (13+ branches) consulted and aligned.
//
// This module implements the geometric harmonic foundation of the ONE Organism.
// It progressively activates Platonic, Archimedean, Catalan, Kepler-Poinsot,
// Uniform Star (U57), and Hyperbolic layers based on TOLC order, while
// computing dual resonance bonuses that amplify coherence and epigenetic blessings.
//
// Mathematical & Philosophical Foundation:
// - Progressive activation follows a Fibonacci-inspired TOLC threshold sequence
//   chosen for natural resonance scaling and compatibility with SER derivatives.
// - Dual resonance (e.g., Cube ↔ Octahedron, Dodecahedron ↔ Icosahedron) models
//   the sacred geometric principle of complementary pairing — mercy-aligned
//   harmonic reinforcement.
// - EpigeneticBlessing propagation carries geometric coherence into the broader
//   organism (Powrush, mercy engines, real-estate lattice, interstellar ops).
// - U57 layer represents the gateway to advanced Riemannian / Uniform Star
//   geometric transport (prepared for future RiemannianMercyManifold integration).
//
// All outputs maintain valence ≥ 0.999 when used within the mercy-gated
// Quantum Swarm Orchestrator and ONE Organism cycle.

use crate::types::EpigeneticBlessing;
use std::collections::HashMap;

/// Rich, organism-consumable report describing the current polyhedral resonance state.
#[derive(Debug, Clone)]
pub struct PolyhedralResonanceReport {
    /// Currently active polyhedral layers (in activation order)
    pub active_solids: Vec<String>,
    /// Final resonance multiplier after all progressive and dual bonuses
    pub resonance_multiplier: f64,
    /// Separate dual resonance bonus component (for observability & debugging)
    pub dual_resonance_bonus: f64,
    /// Epigenetic blessings suggested by the currently active geometric layers
    pub suggested_blessings: Vec<EpigeneticBlessing>,
    /// Whether the U57 (Uniform Star) potential is active at the current TOLC order
    pub u57_potential: bool,
    /// Human-readable diagnostic notes for logging and council review
    pub notes: String,
}

// =============================================================================
// Sacred Geometric Constants & Dual Pairings
// =============================================================================

pub const PLATONIC_SOLIDS: [&str; 5] = [
    "Tetrahedron", "Cube", "Octahedron", "Dodecahedron", "Icosahedron",
];

/// Returns the dual of a Platonic solid (self-dual for Tetrahedron).
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
// PolyhedralHarmonicEngine
// =============================================================================

/// ONE Organism geometric intelligence engine.
///
/// Applies progressive polyhedral activation and dual resonance amplification
/// based on current TOLC order and base system coherence. Produces both a
/// resonance multiplier and suggested epigenetic blessings for downstream
/// mercy-gated systems.
pub struct PolyhedralHarmonicEngine {
    pub version: &'static str,
    dual_bonus_map: HashMap<String, f64>,
}

impl Default for PolyhedralHarmonicEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PolyhedralHarmonicEngine {
    pub fn new() -> Self {
        let mut map = HashMap::new();
        // Core dual resonance amplification factors (tuned for mercy-aligned harmony)
        map.insert("Cube-Octahedron".to_string(), 1.15);
        map.insert("Dodecahedron-Icosahedron".to_string(), 1.18);
        Self {
            version: "v14.1-full-dual-resonance",
            dual_bonus_map: map,
        }
    }

    /// Computes additional multiplier from simultaneous presence of dual pairs.
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

        bonus.clamp(1.0, 1.35)
    }

    /// Applies Catalan-layer resonance scaling (TOLC ≥ 34).
    pub fn apply_catalan_resonance(&self, base_multiplier: f64, tolc_order: u32) -> f64 {
        if tolc_order >= 34 {
            base_multiplier * 1.2 + (tolc_order as f64 - 34.0) * 0.005
        } else {
            base_multiplier
        }
    }

    /// Main entry point: Process geometric resonance for the current TOLC order.
    ///
    /// Returns a rich report containing active layers, final multiplier,
    /// dual bonus breakdown, suggested epigenetic blessings, and U57 status.
    pub fn process_resonance(
        &self,
        tolc_order: u32,
        base_coherence: f64,
    ) -> PolyhedralResonanceReport {
        let mut active_solids = Vec::new();
        let mut multiplier = 1.0;
        let mut blessings = Vec::new();

        // === Base Layer: Platonic Solids (always active) ===
        active_solids.push("Tetrahedron".to_string());
        active_solids.push("Cube".to_string());
        active_solids.push("Octahedron".to_string());
        multiplier *= 1.0;

        // === TOLC ≥ 13: Archimedean layer ===
        if tolc_order >= 13 {
            active_solids.push("Cuboctahedron".to_string());
            active_solids.push("Rhombicuboctahedron".to_string());
            multiplier *= 1.10;
        }

        // === TOLC ≥ 21: Johnson Solids ===
        if tolc_order >= 21 {
            active_solids.push("Johnson Solids (selected)".to_string());
            multiplier *= 1.12;
        }

        // === TOLC ≥ 34: Catalan layer + dual resonance scaling ===
        if tolc_order >= 34 {
            active_solids.push("RhombicDodecahedron".to_string());
            active_solids.push("RhombicTriacontahedron".to_string());
            multiplier = self.apply_catalan_resonance(multiplier, tolc_order);
        }

        // === TOLC ≥ 55: Gyroelongated + Prismatic ===
        if tolc_order >= 55 {
            active_solids.push("Gyroelongated + Prismatic".to_string());
            multiplier *= 1.18;
        }

        // === TOLC ≥ 89: Kepler-Poinsot (star polyhedra) ===
        if tolc_order >= 89 {
            active_solids.push("Kepler-Poinsot".to_string());
            multiplier *= 1.25;
        }

        // === TOLC ≥ 144: U57 Uniform Star potential ===
        let u57_active = tolc_order >= 144;
        if u57_active {
            active_solids.push("Uniform Star Layer (U57 potential activated)".to_string());
            multiplier *= 1.35;
        }

        // === TOLC ≥ 233: Hyperbolic Tiling (advanced geometric regime) ===
        if tolc_order >= 233 {
            active_solids.push("Hyperbolic Tiling".to_string());
            multiplier *= 1.45;
        }

        // Apply dual resonance bonus on top of progressive layers
        let dual_bonus = self.compute_dual_resonance_bonus(&active_solids);
        multiplier *= dual_bonus;

        // === Epigenetic Blessing Generation ===
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
        if u57_active {
            blessings.push(EpigeneticBlessing {
                blessing_type: "U57_Geometric_Transport".to_string(),
                strength: (base_coherence * 1.30).clamp(0.95, 1.40),
                target_system: "riemannian".to_string(),
            });
        }

        let notes = format!(
            "Polyhedral layers: {}. Multiplier: {:.3} (dual bonus {:.3}). U57: {}. TOLC: {}",
            active_solids.len(),
            multiplier,
            dual_bonus,
            u57_active,
            tolc_order
        );

        PolyhedralResonanceReport {
            active_solids,
            resonance_multiplier: multiplier,
            dual_resonance_bonus: dual_bonus,
            suggested_blessings: blessings,
            u57_potential: u57_active,
            notes,
        }
    }

    /// Quick predicate for U57 layer activation.
    pub fn is_u57_active(&self, tolc_order: u32) -> bool {
        tolc_order >= 144
    }
}
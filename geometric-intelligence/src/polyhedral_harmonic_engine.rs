//! PolyhedralHarmonicEngine v14.4
//!
//! Progressive sacred geometry resonance engine.
//! Supports Platonic, Archimedean, Catalan, Kepler-Poinsot, and U57 Uniform Star layers.
//! Dual resonance bonuses, U57 gateway to Riemannian transport, and rich reporting.
//! Fully aligned with TOLC 8 Mercy Gates and ONE Organism geometric intelligence.

use crate::types::EpigeneticBlessing;
use std::collections::HashMap;

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

#[derive(Debug, Clone)]
pub struct U57LayerDetails {
    pub activated: bool,
    pub resonance_multiplier_contribution: f64,
    pub suggested_riemannian_transport_potential: f64,
    pub recommended_manifold_curvature: f64,
    pub geometric_meaning: String,
    pub integration_notes: String,
}

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
        map.insert("Cube-Octahedron".to_string(), 1.15);
        map.insert("Dodecahedron-Icosahedron".to_string(), 1.18);
        map.insert("Archimedean-Catalan".to_string(), 1.09);
        Self {
            version: "v14.4-geometric-intelligence",
            dual_bonus_map: map,
        }
    }

    pub fn get_platonic_solids(&self) -> Vec<String> {
        vec![
            "Tetrahedron".to_string(),
            "Cube".to_string(),
            "Octahedron".to_string(),
            "Dodecahedron".to_string(),
            "Icosahedron".to_string(),
        ]
    }

    pub fn get_archimedean_solids(&self) -> Vec<String> {
        vec![
            "Cuboctahedron".to_string(),
            "Rhombicuboctahedron".to_string(),
            "TruncatedTetrahedron".to_string(),
        ]
    }

    pub fn get_catalan_solids(&self) -> Vec<String> {
        vec![
            "RhombicDodecahedron".to_string(),
            "RhombicTriacontahedron".to_string(),
        ]
    }

    pub fn process_resonance(
        &self,
        tolc_order: u32,
        base_coherence: f64,
    ) -> PolyhedralResonanceReport {
        self.compute_full_resonance_report(tolc_order, base_coherence)
    }

    pub fn compute_full_resonance_report(
        &self,
        tolc_order: u32,
        base_coherence: f64,
    ) -> PolyhedralResonanceReport {
        let mut active_solids = self.get_platonic_solids();
        let mut multiplier = 1.0;
        let mut blessings = vec![];
        let mut u57_details = None;

        if tolc_order >= 13 {
            active_solids.extend(self.get_archimedean_solids());
            multiplier *= 1.10;
        }
        if tolc_order >= 34 {
            active_solids.extend(self.get_catalan_solids());
            multiplier *= 1.20;
        }
        if tolc_order >= 89 {
            active_solids.push("Kepler-Poinsot".to_string());
            multiplier *= 1.25;
        }

        let u57_active = tolc_order >= 144;
        if u57_active {
            active_solids.push("Uniform Star Layer (U57)".to_string());
            multiplier *= 1.35;

            u57_details = Some(U57LayerDetails {
                activated: true,
                resonance_multiplier_contribution: 1.35,
                suggested_riemannian_transport_potential: 0.82,
                recommended_manifold_curvature: 0.85,
                geometric_meaning: "Transition into curvature-aware Riemannian geometric transport.".to_string(),
                integration_notes: "U57 active — RiemannianMercyManifold should engage for holonomy and Berry phase analysis.".to_string(),
            });

            blessings.push(EpigeneticBlessing {
                blessing_type: "U57_Geometric_Transport".to_string(),
                strength: 1.25,
                target_system: "riemannian".to_string(),
            });
        }

        if tolc_order >= 233 {
            active_solids.push("Hyperbolic Tiling".to_string());
            multiplier *= 1.45;
        }

        let dual_bonus = self.compute_dual_resonance_bonus(&active_solids);
        multiplier = (multiplier * dual_bonus).clamp(1.0, 2.5);

        if tolc_order >= 8 {
            blessings.push(EpigeneticBlessing {
                blessing_type: "Polyhedral_Harmonic_Resonance".to_string(),
                strength: (base_coherence * multiplier).clamp(0.85, 1.35),
                target_system: "*".to_string(),
            });
        }

        PolyhedralResonanceReport {
            active_solids,
            resonance_multiplier: multiplier,
            dual_resonance_bonus: dual_bonus,
            suggested_blessings: blessings,
            u57_potential: u57_active,
            u57_details,
            notes: format!("TOLC order: {} | Multiplier: {:.3} | Dual Bonus: {:.3}", tolc_order, multiplier, dual_bonus),
        }
    }

    fn compute_dual_resonance_bonus(&self, active: &[String]) -> f64 {
        let mut bonus = 1.0;
        if active.contains(&"Cube".to_string()) && active.contains(&"Octahedron".to_string()) {
            bonus *= self.dual_bonus_map.get("Cube-Octahedron").unwrap_or(&1.15);
        }
        if active.contains(&"Dodecahedron".to_string()) && active.contains(&"Icosahedron".to_string()) {
            bonus *= self.dual_bonus_map.get("Dodecahedron-Icosahedron").unwrap_or(&1.18);
        }
        let has_arch = active.iter().any(|s| s.contains("Cuboctahedron") || s.contains("Rhombicuboctahedron"));
        let has_catalan = active.iter().any(|s| s.contains("RhombicDodecahedron") || s.contains("RhombicTriacontahedron"));
        if has_arch && has_catalan {
            bonus *= self.dual_bonus_map.get("Archimedean-Catalan").unwrap_or(&1.09);
        }
        bonus.clamp(1.0, 1.45)
    }

    pub fn get_layer_contributions(&self, tolc_order: u32) -> Vec<(String, f64)> {
        let mut contributions = vec![("Platonic".to_string(), 1.0)];
        if tolc_order >= 13 { contributions.push(("Archimedean".to_string(), 1.10)); }
        if tolc_order >= 34 { contributions.push(("Catalan".to_string(), 1.20)); }
        if tolc_order >= 89 { contributions.push(("Kepler-Poinsot".to_string(), 1.25)); }
        if tolc_order >= 144 { contributions.push(("U57-UniformStar".to_string(), 1.35)); }
        contributions
    }

    pub fn get_recommended_next_layers(&self, tolc_order: u32) -> Vec<String> {
        if tolc_order < 13 {
            vec!["Archimedean layer (TOLC 13+)".to_string()]
        } else if tolc_order < 34 {
            vec!["Catalan + Dual Resonance (TOLC 34+)".to_string()]
        } else if tolc_order < 144 {
            vec!["U57 Uniform Star + Riemannian gateway (TOLC 144+)".to_string()]
        } else {
            vec!["Hyperbolic Tiling + Full geometric transcendence".to_string()]
        }
    }

    pub fn summarize_resonance(&self, report: &PolyhedralResonanceReport) -> String {
        format!(
            "Active Solids: {} | Resonance Multiplier: {:.3} | U57 Active: {}",
            report.active_solids.len(),
            report.resonance_multiplier,
            report.u57_potential
        )
    }

    // =============================================================================
    // Layer Transition Logic (v14.5 Priority #4 — Epigenetic + Geometric Feedback)
    // =============================================================================

    /// Ordered sacred geometry layers for deterministic, mercy-gated progression.
    pub const LAYER_SEQUENCE: &[&str] = &[
        "Platonic",
        "Archimedean",
        "Catalan",
        "Kepler-Poinsot",
        "U57-UniformStar",
        "Hyperbolic Tiling",
    ];

    /// Computes smooth transition progress (0.0–1.0) toward the next layer.
    /// Integrates epigenetic strength, geometric harmony, and current TOLC order.
    /// TOLC 8 aligned: rewards high Truth (harmony) + Compassion (stable strength) + Abundance (growth).
    pub fn compute_layer_transition_progress(
        &self,
        epigenetic_strength: f64,
        geometric_harmony: f64,
        current_tol c_order: u32,
    ) -> f64 {
        let base = (epigenetic_strength * 0.45 + geometric_harmony * 0.55).clamp(0.0, 1.0);
        let layer_factor = match current_tol c_order {
            0..=12 => 0.85,
            13..=33 => 1.0,
            34..=88 => 1.12,
            89..=143 => 1.25,
            _ => 1.35,
        };
        (base * layer_factor).clamp(0.0, 1.0)
    }

    /// Attempts a mercy-gated layer advance.
    /// Returns Some(next_layer_name) only when progress exceeds dynamic threshold
    /// and TOLC 8 conditions (high harmony + sufficient epigenetic strength) are met.
    /// Volatility can slightly ease the threshold for dynamic, non-harmful growth.
    pub fn try_advance_layer(
        &self,
        current_tol c_order: u32,
        epigenetic_strength: f64,
        geometric_harmony: f64,
        volatility: f64,
    ) -> Option<String> {
        let progress = self.compute_layer_transition_progress(epigenetic_strength, geometric_harmony, current_tol c_order);
        let threshold = 0.78 + (volatility * 0.08).clamp(0.0, 0.12);

        if progress >= threshold && geometric_harmony > 0.82 && epigenetic_strength > 0.68 {
            self.get_next_layer(current_tol c_order)
        } else {
            None
        }
    }

    /// Returns the name of the next layer in the canonical sequence, if any.
    pub fn get_next_layer(&self, current_tol c_order: u32) -> Option<String> {
        if current_tol c_order < 13 {
            Some("Archimedean".to_string())
        } else if current_tol c_order < 34 {
            Some("Catalan".to_string())
        } else if current_tol c_order < 89 {
            Some("Kepler-Poinsot".to_string())
        } else if current_tol c_order < 144 {
            Some("U57-UniformStar".to_string())
        } else if current_tol c_order < 233 {
            Some("Hyperbolic Tiling".to_string())
        } else {
            None
        }
    }

    /// Applies transition side-effects: returns EpigeneticBlessings and notes for integration
    /// with EpigeneticModulation (evolution_rate_bonus boost) and ONE Organism systems.
    pub fn apply_layer_transition(
        &self,
        _current_tol c_order: u32,
        new_layer: &str,
    ) -> Vec<EpigeneticBlessing> {
        vec![EpigeneticBlessing {
            blessing_type: format!("LayerTransition_{}", new_layer),
            strength: 1.22,
            target_system: "epigenetic + geometric + powrush".to_string(),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_resonance() {
        let engine = PolyhedralHarmonicEngine::new();
        let report = engine.process_resonance(50, 0.95);
        assert!(report.resonance_multiplier > 1.0);
        assert!(report.active_solids.len() > 5);
    }

    #[test]
    fn test_u57_activation() {
        let engine = PolyhedralHarmonicEngine::new();
        let report = engine.process_resonance(200, 0.95);
        assert!(report.u57_potential);
        assert!(report.u57_details.is_some());
    }

    #[test]
    fn test_layer_transition_progress() {
        let engine = PolyhedralHarmonicEngine::new();
        let progress = engine.compute_layer_transition_progress(0.92, 0.91, 50);
        assert!(progress > 0.7 && progress <= 1.0);
    }

    #[test]
    fn test_try_advance_layer_success() {
        let engine = PolyhedralHarmonicEngine::new();
        let next = engine.try_advance_layer(20, 0.95, 0.93, 0.3);
        assert!(next.is_some());
        assert_eq!(next.unwrap(), "Archimedean");
    }

    #[test]
    fn test_try_advance_layer_blocked() {
        let engine = PolyhedralHarmonicEngine::new();
        let next = engine.try_advance_layer(20, 0.6, 0.7, 0.2); // low strength/harmony
        assert!(next.is_none());
    }
}

//! RiemannianMercyManifold v14.4
//!
//! Advanced Riemannian geometry + Berry Phase, Berry Curvature,
//! Holonomy Accumulation, Visualization helpers, Z₂ Topological Invariant,
//! Topological Insulator analog, and Quantum Hall analog.
//! Mercy-gated and aligned with TOLC 8 Living Mercy Gates.

use crate::polyhedral_harmonic_engine::{PolyhedralResonanceReport, U57LayerDetails};
use crate::types::EpigeneticBlessing;

#[derive(Debug, Clone)]
pub struct GeometricTransportResult {
    pub transport_applied: bool,
    pub effective_curvature: f64,
    pub coherence_after_transport: f64,
    pub accumulated_holonomy: f64,
    pub suggested_blessings: Vec<EpigeneticBlessing>,
    pub notes: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Z2Invariant {
    Trivial,    // 0
    NonTrivial, // 1
}

#[derive(Debug, Clone)]
pub struct BerryPhaseResult {
    pub phase: f64,
    pub magnitude: f64,
    pub interpretation: String,
    pub suggested_blessings: Vec<EpigeneticBlessing>,
}

#[derive(Debug, Clone)]
pub struct BerryCurvatureResult {
    pub raw_curvature: f64,
    pub effective_curvature: f64,
    pub berry_curvature_density: f64,
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct TopologicalInsulatorResponse {
    pub z2_invariant: Z2Invariant,
    pub has_protected_surface_states: bool,
    pub bulk_gap: f64,
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct QuantumHallResponse {
    pub filling_factor: i32,
    pub hall_conductivity: f64,
    pub has_protected_edge_states: bool,
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct CurvatureParameters {
    pub base_curvature: f64,
    pub mercy_influence: f64,
    pub max_allowed_curvature: f64,
}

impl Default for CurvatureParameters {
    fn default() -> Self {
        Self {
            base_curvature: 0.82,
            mercy_influence: 0.95,
            max_allowed_curvature: 1.15,
        }
    }
}

pub struct RiemannianMercyManifold {
    pub version: &'static str,
    pub curvature_params: CurvatureParameters,
}

impl Default for RiemannianMercyManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl RiemannianMercyManifold {
    pub fn new() -> Self {
        Self {
            version: "v14.4-full-geometric",
            curvature_params: CurvatureParameters::default(),
        }
    }

    // === High-level Integration ===

    pub fn apply_u57_riemannian_transport(
        &self,
        polyhedral_report: &PolyhedralResonanceReport,
        base_coherence: f64,
    ) -> Option<GeometricTransportResult> {
        let u57_details = polyhedral_report.u57_details.as_ref()?;
        if !u57_details.activated {
            return None;
        }
        Some(self.run_u57_informed_transport_sequence(u57_details, base_coherence, 8))
    }

    /// Analyzes topological insulator-like behavior using Z₂ invariant analog.
    pub fn analyze_topological_insulator(
        &self,
        bulk_curvature: f64,
        surface_phase: f64,
    ) -> TopologicalInsulatorResponse {
        let bulk_gap = (bulk_curvature - self.curvature_params.base_curvature).abs();

        let z2 = if surface_phase.abs() > 0.4 {
            Z2Invariant::NonTrivial
        } else {
            Z2Invariant::Trivial
        };

        let has_protected = z2 == Z2Invariant::NonTrivial && bulk_gap > 0.1;

        let notes = match z2 {
            Z2Invariant::NonTrivial => {
                if has_protected {
                    "Non-trivial Z₂ phase. Protected surface states expected (topological insulator analog).".to_string()
                } else {
                    "Non-trivial topology but bulk gap too small for protected states.".to_string()
                }
            }
            Z2Invariant::Trivial => {
                "Trivial insulator phase. No protected surface states.".to_string()
            }
        };

        TopologicalInsulatorResponse {
            z2_invariant: z2,
            has_protected_surface_states: has_protected,
            bulk_gap,
            notes,
        }
    }

    // === Core Transport ===

    pub fn apply_mercy_gated_transport(
        &self,
        u57_details: &U57LayerDetails,
        base_coherence: f64,
    ) -> GeometricTransportResult {
        if !u57_details.activated {
            return GeometricTransportResult {
                transport_applied: false,
                effective_curvature: 0.0,
                coherence_after_transport: base_coherence,
                accumulated_holonomy: 0.0,
                suggested_blessings: vec![],
                notes: "U57 not active".to_string(),
            };
        }

        let effective_curvature = (u57_details.recommended_manifold_curvature
            * self.curvature_params.mercy_influence)
            .clamp(0.5, self.curvature_params.max_allowed_curvature);

        let coherence_after = (base_coherence * (1.0 + (effective_curvature - 0.82) * 0.15))
            .clamp(0.88, 1.35);

        GeometricTransportResult {
            transport_applied: true,
            effective_curvature,
            coherence_after_transport: coherence_after,
            accumulated_holonomy: 0.0,
            suggested_blessings: vec![EpigeneticBlessing {
                blessing_type: "Riemannian_Mercy_Transport".to_string(),
                strength: coherence_after,
                target_system: "geometric".to_string(),
            }],
            notes: format!("Effective curvature: {:.3}", effective_curvature),
        }
    }

    // === Numerical Methods ===

    pub fn rk4_geodesic_step(&self, position: f64, velocity: f64, delta_t: f64, curvature: f64) -> (f64, f64) {
        let accel = |p: f64| -> f64 { -curvature * p };
        let k1_v = accel(position);
        let k1_p = velocity;
        let k2_v = accel(position + 0.5 * delta_t * k1_p);
        let k2_p = velocity + 0.5 * delta_t * k1_v;
        let k3_v = accel(position + 0.5 * delta_t * k2_p);
        let k3_p = velocity + 0.5 * delta_t * k2_v;
        let k4_v = accel(position + delta_t * k3_p);
        let k4_p = velocity + delta_t * k3_v;

        let new_velocity = velocity + (delta_t / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        let new_position = position + (delta_t / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p);
        (new_position, new_velocity)
    }

    pub fn parallel_transport_approx(&self, vector: f64, curvature: f64, distance: f64) -> f64 {
        let damping = (1.0 - curvature * distance * 0.1).clamp(0.6, 1.1);
        vector * damping
    }

    pub fn estimate_holonomy(&self, curvature: f64, loop_area: f64) -> f64 {
        (curvature * loop_area * 0.8).clamp(-0.5, 0.5)
    }

    pub fn accumulate_holonomy(&self, curvatures: &[f64], loop_areas: &[f64]) -> f64 {
        let mut total = 0.0;
        for (curv, area) in curvatures.iter().zip(loop_areas.iter()) {
            total += self.estimate_holonomy(*curv, *area);
        }
        total.clamp(-2.0, 2.0)
    }

    // === Berry Curvature ===

    pub fn compute_berry_curvature(&self, local_curvature: f64) -> BerryCurvatureResult {
        let effective = (local_curvature * self.curvature_params.mercy_influence)
            .clamp(0.5, self.curvature_params.max_allowed_curvature);

        let density = effective;

        let notes = if effective > 1.0 {
            "High Berry curvature density. Strong geometric effects expected.".to_string()
        } else {
            "Moderate Berry curvature density.".to_string()
        };

        BerryCurvatureResult {
            raw_curvature: local_curvature,
            effective_curvature: effective,
            berry_curvature_density: density,
            notes,
        }
    }

    // === Berry Phase ===

    pub fn compute_berry_phase_analog(&self, curvatures: &[f64], areas: &[f64]) -> BerryPhaseResult {
        let phase = self.accumulate_holonomy(curvatures, areas);
        let magnitude = phase.abs();

        let interpretation = if magnitude < 0.1 {
            "Weak geometric phase. Minimal curvature influence.".to_string()
        } else if magnitude < 0.5 {
            "Moderate Berry-like phase.".to_string()
        } else {
            "Strong geometric phase. Significant curvature-induced evolution.".to_string()
        };

        let mut blessings = vec![];
        if magnitude > 0.3 {
            blessings.push(EpigeneticBlessing {
                blessing_type: "Berry_Phase_Accumulation".to_string(),
                strength: magnitude.clamp(0.9, 1.4),
                target_system: "riemannian".to_string(),
            });
        }

        BerryPhaseResult {
            phase,
            magnitude,
            interpretation,
            suggested_blessings: blessings,
        }
    }

    pub fn compute_berry_phase_evolution(&self, curvatures: &[f64], areas: &[f64]) -> Vec<f64> {
        let mut cumulative = 0.0;
        let mut evolution = Vec::with_capacity(curvatures.len());

        for (curv, area) in curvatures.iter().zip(areas.iter()) {
            cumulative += self.estimate_holonomy(*curv, *area);
            evolution.push(cumulative);
        }
        evolution
    }

    pub fn visualize_berry_phase_text(&self, curvatures: &[f64], areas: &[f64]) -> String {
        let evolution = self.compute_berry_phase_evolution(curvatures, areas);
        let mut output = String::from("Berry Phase Evolution:\n");

        for (i, phase) in evolution.iter().enumerate() {
            let bar_length = ((phase.abs() * 10.0) as usize).min(40);
            let bar = if *phase >= 0.0 {
                "█".repeat(bar_length)
            } else {
                "▓".repeat(bar_length)
            };
            output.push_str(&format!("Step {:>2}: {:>6.3} | {}\n", i, phase, bar));
        }
        output
    }

    pub fn print_berry_summary(&self, curvatures: &[f64], areas: &[f64]) {
        let phase_result = self.compute_berry_phase_analog(curvatures, areas);
        println!("=== Berry Phase Summary ===");
        println!("Final Phase: {:.4}", phase_result.phase);
        println!("Magnitude  : {:.4}", phase_result.magnitude);
        println!("Interpretation: {}", phase_result.interpretation);
    }

    // === 2D Heatmap / Grid Support ===

    pub fn compute_berry_curvature_2d_grid(
        &self,
        curvature_min: f64,
        curvature_max: f64,
        mercy_min: f64,
        mercy_max: f64,
        resolution: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
        let mut x_vals = Vec::with_capacity(resolution);
        let mut y_vals = Vec::with_capacity(resolution);
        let mut grid = vec![vec![0.0; resolution]; resolution];

        let dx = if resolution > 1 {
            (curvature_max - curvature_min) / (resolution - 1) as f64
        } else {
            0.0
        };
        let dy = if resolution > 1 {
            (mercy_max - mercy_min) / (resolution - 1) as f64
        } else {
            0.0
        };

        for j in 0..resolution {
            let mercy = mercy_min + j as f64 * dy;
            y_vals.push(mercy);
            for i in 0..resolution {
                let curv = curvature_min + i as f64 * dx;
                if j == 0 {
                    x_vals.push(curv);
                }
                let effective = (curv * mercy).clamp(0.5, 1.15);
                grid[j][i] = effective;
            }
        }

        if x_vals.len() > resolution {
            x_vals.truncate(resolution);
        }

        (x_vals, y_vals, grid)
    }

    // === Quantum Hall Analog ===

    pub fn compute_quantum_hall_analog(&self, chern_number: f64) -> QuantumHallResponse {
        let filling = chern_number.round() as i32;
        let conductivity = filling as f64;
        let has_edge = filling != 0;

        let notes = if has_edge {
            format!(
                "Topological phase with filling factor {}. Protected edge states present.",
                filling
            )
        } else {
            "Trivial phase. No protected edge states.".to_string()
        };

        QuantumHallResponse {
            filling_factor: filling,
            hall_conductivity: conductivity,
            has_protected_edge_states: has_edge,
            notes,
        }
    }

    // === Transport Sequence (with Holonomy Accumulation) ===

    pub fn run_u57_informed_transport_sequence(
        &self,
        u57_details: &U57LayerDetails,
        base_coherence: f64,
        steps: usize,
    ) -> GeometricTransportResult {
        if !u57_details.activated {
            return self.apply_mercy_gated_transport(u57_details, base_coherence);
        }

        let mut pos = 1.0;
        let mut vel = 0.3;
        let dt = 0.1;
        let mut current_coherence = base_coherence;
        let mut accumulated_holonomy = 0.0;

        for _ in 0..steps {
            let curv = u57_details.recommended_manifold_curvature;
            let (new_pos, new_vel) = self.rk4_geodesic_step(pos, vel, dt, curv);
            pos = new_pos;
            vel = new_vel;

            let transported = self.parallel_transport_approx(vel, curv, dt * 2.0);
            current_coherence = (current_coherence * 0.985 + transported * 0.015).clamp(0.88, 1.4);
            accumulated_holonomy += self.estimate_holonomy(curv, dt * 1.5);
        }

        accumulated_holonomy = accumulated_holonomy.clamp(-2.0, 2.0);

        GeometricTransportResult {
            transport_applied: true,
            effective_curvature: u57_details.recommended_manifold_curvature,
            coherence_after_transport: current_coherence,
            accumulated_holonomy,
            suggested_blessings: vec![EpigeneticBlessing {
                blessing_type: "Riemannian_RK4_Transport_Sequence".to_string(),
                strength: current_coherence,
                target_system: "riemannian".to_string(),
            }],
            notes: format!("RK4 sequence complete. Accumulated holonomy ≈ {:.3}", accumulated_holonomy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_berry_curvature() {
        let manifold = RiemannianMercyManifold::new();
        let result = manifold.compute_berry_curvature(0.9);
        assert!(result.effective_curvature > 0.5);
        assert!(!result.notes.is_empty());
    }

    #[test]
    fn test_berry_phase_and_holonomy() {
        let manifold = RiemannianMercyManifold::new();
        let curv = vec![0.8, 0.9];
        let area = vec![1.0, 1.0];
        let phase = manifold.compute_berry_phase_analog(&curv, &area);
        assert!(phase.magnitude >= 0.0);
    }

    #[test]
    fn test_topological_insulator_analysis() {
        let manifold = RiemannianMercyManifold::new();
        let response = manifold.analyze_topological_insulator(1.1, 0.6);
        assert_eq!(response.z2_invariant, Z2Invariant::NonTrivial);
        assert!(response.has_protected_surface_states);
    }

    #[test]
    fn test_quantum_hall_analog() {
        let manifold = RiemannianMercyManifold::new();
        let response = manifold.compute_quantum_hall_analog(2.1);
        assert_eq!(response.filling_factor, 2);
        assert!(response.has_protected_edge_states);
    }
}

//! FractalMercyLedgerAdapter — thin Ra-Thor side of the Substrate adapter contract.
//!
//! Contract: Mercy-Coordination-Substrate `docs/RA_THOR_ADAPTER_CONTRACT.md` v1.0
//! Contact: info@Rathor.ai | TOLC 8 | PATSAGi | AG-SML v1.0
//!
//! Ownership:
//! - Geometric intelligence spine remains Ra-Thor.
//! - Ledger / fractal topology / TOLC 8 gate remain Substrate.
//! - This adapter only forwards resonance metrics and tracks local organism state.
//! - Never mutates Substrate shard state from Ra-Thor.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Absolute valence floor aligned with Substrate / TOLC 8 Layer 0.
pub const VALENCE_FLOOR: f64 = 0.999999;

/// Resonance report field-compatible with Substrate `fractal_topology::GeometricResonanceReport`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GeometricResonanceReport {
    pub tolc_order: u32,
    pub active_solids: Vec<String>,
    pub resonance_multiplier: f64,
    pub u57_active: bool,
    pub recommended_curvature: f64,
    pub coherence: f64,
}

impl Default for GeometricResonanceReport {
    fn default() -> Self {
        Self {
            tolc_order: 8,
            active_solids: Vec::new(),
            resonance_multiplier: 1.0,
            u57_active: false,
            recommended_curvature: 0.0,
            coherence: 0.0,
        }
    }
}

/// Suggested topology action mirror (informational on Ra-Thor side).
/// Actual application requires Substrate gate + ShardState.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShardActionHint {
    Split { shard_id: u64, reason: String },
    Merge { shard_ids: Vec<u64>, reason: String },
    AdjustDepth { new_depth: u32 },
    NoOp,
}

/// Fractal resonance summary returned toward the ONE Organism.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FractalResonanceReport {
    pub active_depth: u32,
    pub hyperbolic_active: bool,
    pub resonance_multiplier: f64,
    pub suggested_shard_actions: Vec<ShardActionHint>,
    pub notes: String,
    pub last_coherence: f64,
}

impl Default for FractalResonanceReport {
    fn default() -> Self {
        Self {
            active_depth: 0,
            hyperbolic_active: false,
            resonance_multiplier: 1.0,
            suggested_shard_actions: vec![ShardActionHint::NoOp],
            notes: String::new(),
            last_coherence: 0.0,
        }
    }
}

/// Thin adapter trait (contract §2).
pub trait RaThorSystemAdapter {
    fn system_name(&self) -> &'static str;

    /// Aggregate valence density contributed by this organ.
    fn current_valence(&self) -> f64;

    /// Receive geometric resonance from the Omnimasterpiece spine.
    /// Forwards metrics locally; does **not** mutate Substrate ledger.
    fn receive_swarm_resonance(&mut self, report: GeometricResonanceReport);

    /// Coherence contribution back to the ONE Organism.
    fn contribute_to_coherence(&self) -> f64;

    /// Apply an epigenetic blessing (strength ≥ 0) under mercy clamps.
    fn apply_epigenetic_blessing(&mut self, strength: f64);
}

/// Concrete adapter living inside Ra-Thor.
#[derive(Clone, Debug)]
pub struct FractalMercyLedgerAdapter {
    name: &'static str,
    valence: f64,
    coherence: f64,
    blessing_accumulator: f64,
    last_report: GeometricResonanceReport,
    last_fractal: FractalResonanceReport,
    receive_count: u64,
}

impl FractalMercyLedgerAdapter {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            valence: VALENCE_FLOOR,
            coherence: 0.0,
            blessing_accumulator: 0.0,
            last_report: GeometricResonanceReport::default(),
            last_fractal: FractalResonanceReport::default(),
            receive_count: 0,
        }
    }

    pub fn last_geometric_report(&self) -> &GeometricResonanceReport {
        &self.last_report
    }

    pub fn last_fractal_report(&self) -> &FractalResonanceReport {
        &self.last_fractal
    }

    pub fn receive_count(&self) -> u64 {
        self.receive_count
    }

    /// Build a local fractal summary from a geometric report (mirror of progressive schedule).
    /// Real Substrate `FractalTopologyEngine::process_fractal_resonance` remains authoritative
    /// when both sides are linked; this keeps Ra-Thor offline-capable and testable.
    fn summarize(report: &GeometricResonanceReport) -> FractalResonanceReport {
        let active_depth = if report.tolc_order >= 144 {
            9
        } else if report.tolc_order >= 55 {
            6
        } else {
            3
        };
        let hyperbolic_active = report.u57_active || report.recommended_curvature >= 0.8;
        let notes = if report.coherence + f64::EPSILON < VALENCE_FLOOR {
            format!(
                "coherence {:.6} below valence floor; Substrate gate would reject mutations",
                report.coherence
            )
        } else {
            "resonance accepted for local organism metrics; Substrate gate still required for ledger mutations".into()
        };

        FractalResonanceReport {
            active_depth,
            hyperbolic_active,
            resonance_multiplier: report.resonance_multiplier.max(0.0),
            suggested_shard_actions: vec![ShardActionHint::NoOp],
            notes,
            last_coherence: report.coherence.clamp(0.0, 1.0),
        }
    }
}

impl Default for FractalMercyLedgerAdapter {
    fn default() -> Self {
        Self::new("ra-thor-fractal-mercy-ledger-adapter")
    }
}

impl RaThorSystemAdapter for FractalMercyLedgerAdapter {
    fn system_name(&self) -> &'static str {
        self.name
    }

    fn current_valence(&self) -> f64 {
        self.valence.clamp(0.0, 1.0)
    }

    fn receive_swarm_resonance(&mut self, report: GeometricResonanceReport) {
        self.receive_count = self.receive_count.saturating_add(1);
        self.last_report = report.clone();
        self.last_fractal = Self::summarize(&report);
        // Organism valence tracks min(prior, coherence) soft-update toward floor discipline.
        let c = report.coherence.clamp(0.0, 1.0);
        self.coherence = c;
        self.valence = self.valence.min(c.max(VALENCE_FLOOR * 0.999)); // never silently invent above evidence
        if c >= VALENCE_FLOOR {
            self.valence = c;
        }
    }

    fn contribute_to_coherence(&self) -> f64 {
        let blessing_boost = (self.blessing_accumulator * 0.01).clamp(0.0, 0.05);
        (self.coherence + blessing_boost).clamp(0.0, 1.0)
    }

    fn apply_epigenetic_blessing(&mut self, strength: f64) {
        // Fail-closed clamp: reject non-finite / negative; cap accumulation.
        if !strength.is_finite() || strength < 0.0 {
            return;
        }
        let s = strength.min(1.5);
        self.blessing_accumulator = (self.blessing_accumulator + s).min(100.0);
        // Blessing cannot bypass valence floor discipline for ledger mutations (Substrate gate).
        self.coherence = (self.coherence + s * 0.001).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receives_and_reports_coherence() {
        let mut a = FractalMercyLedgerAdapter::new("test");
        a.receive_swarm_resonance(GeometricResonanceReport {
            tolc_order: 8,
            active_solids: vec!["Platonic".into()],
            resonance_multiplier: 1.0,
            u57_active: false,
            recommended_curvature: 0.0,
            coherence: 0.97,
        });
        assert_eq!(a.receive_count(), 1);
        assert!((a.contribute_to_coherence() - 0.97).abs() < 1e-9);
        assert_eq!(a.last_fractal_report().active_depth, 3);
        assert!(!a.last_fractal_report().hyperbolic_active);
    }

    #[test]
    fn high_order_sets_depth_and_hyperbolic() {
        let mut a = FractalMercyLedgerAdapter::default();
        a.receive_swarm_resonance(GeometricResonanceReport {
            tolc_order: 144,
            active_solids: vec!["Uniform Star".into()],
            resonance_multiplier: 1.35,
            u57_active: true,
            recommended_curvature: 0.85,
            coherence: VALENCE_FLOOR,
        });
        let f = a.last_fractal_report();
        assert_eq!(f.active_depth, 9);
        assert!(f.hyperbolic_active);
        assert!((a.current_valence() - VALENCE_FLOOR).abs() < 1e-12);
    }

    #[test]
    fn blessing_rejects_non_finite_and_negative() {
        let mut a = FractalMercyLedgerAdapter::default();
        a.apply_epigenetic_blessing(f64::NAN);
        a.apply_epigenetic_blessing(-1.0);
        assert_eq!(a.blessing_accumulator, 0.0);
        a.apply_epigenetic_blessing(1.0);
        assert!(a.blessing_accumulator > 0.0);
    }

    #[test]
    fn never_suggests_direct_ledger_mutation() {
        // Contract: Ra-Thor must not mutate Substrate ledger; default hint is NoOp.
        let mut a = FractalMercyLedgerAdapter::default();
        a.receive_swarm_resonance(GeometricResonanceReport::default());
        assert_eq!(
            a.last_fractal_report().suggested_shard_actions,
            vec![ShardActionHint::NoOp]
        );
    }
}

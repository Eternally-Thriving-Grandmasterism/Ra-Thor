// crates/quantum-swarm-orchestrator/src/adapters/fractal_mercy_ledger_adapter.rs
//
// FractalMercyLedgerAdapter — Thin Ra-Thor side adapter
// Matches the binding contract published in
// Mercy-Coordination-Substrate/docs/RA_THOR_ADAPTER_CONTRACT.md
//
// PATSAGi + Ra-Thor decision: this is the final pure work item that closes
// the geometric intelligence → Substrate fractal topology loop.
//
// Ownership boundary remains absolute:
// - Geometric intelligence lives in Ra-Thor
// - Ledger / PQ / BFT / gated shard mutations live in the Substrate
// - This adapter is the only permitted bridge and lives inside Ra-Thor
//
// AG-SML v1.0

use crate::adapter::RaThorSystemAdapter;
use crate::polyhedral_harmonic_engine::PolyhedralResonanceReport;
use crate::types::{
    EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence,
};

/// Local mirror of the Substrate GeometricResonanceReport shape.
/// (Until a path/git dependency on the Substrate crates is introduced,
///  we keep a pure local equivalent so the adapter remains compilable
///  and the contract is satisfied in form and spirit.)
#[derive(Clone, Debug)]
pub struct GeometricResonanceReport {
    pub tolc_order: u32,
    pub active_solids: Vec<String>,
    pub resonance_multiplier: f64,
    pub u57_active: bool,
    pub recommended_curvature: f64,
    pub coherence: f64,
}

impl From<&PolyhedralResonanceReport> for GeometricResonanceReport {
    fn from(r: &PolyhedralResonanceReport) -> Self {
        let (u57_active, recommended_curvature) = r
            .u57_details
            .as_ref()
            .map(|d| (d.activated, d.recommended_manifold_curvature))
            .unwrap_or((false, 0.0));

        Self {
            tolc_order: 0, // caller should set if known; 0 is safe default
            active_solids: r.active_solids.clone(),
            resonance_multiplier: r.resonance_multiplier,
            u57_active,
            recommended_curvature,
            coherence: r.resonance_multiplier.clamp(0.88, 1.35),
        }
    }
}

/// Thin adapter that satisfies the Substrate RA_THOR_ADAPTER_CONTRACT
/// while also participating in the richer local ONE Organism cycle.
pub struct FractalMercyLedgerAdapter {
    name: &'static str,
    current_valence: Valence,
    last_geometric_report: Option<GeometricResonanceReport>,
    coherence_contribution: f64,
    blessing_count: u64,
}

impl FractalMercyLedgerAdapter {
    pub fn new() -> Self {
        Self {
            name: "FractalMercyLedger",
            current_valence: Valence(0.99999997),
            last_geometric_report: None,
            coherence_contribution: 0.94,
            blessing_count: 0,
        }
    }

    /// Primary contract method: receive a geometric resonance report
    /// produced by the Omnimasterpiece spine and hold it for forwarding
    /// into the Substrate FractalTopologyEngine (when the path dependency
    /// is later activated).
    pub fn receive_geometric_resonance(&mut self, report: GeometricResonanceReport) {
        // Mild valence uplift under high coherence (mercy-gated)
        if report.coherence > 0.95 {
            let boosted = (self.current_valence.value() + 0.000000005).clamp(0.999999, 1.0);
            self.current_valence = Valence(boosted);
        }
        self.coherence_contribution = (report.coherence * 0.98).clamp(0.88, 1.0);
        self.last_geometric_report = Some(report);
    }

    /// Convenience: build GeometricResonanceReport from a local PolyhedralResonanceReport
    /// and receive it in one step.
    pub fn receive_from_polyhedral(
        &mut self,
        poly: &PolyhedralResonanceReport,
        tolc_order: u32,
    ) {
        let mut report = GeometricResonanceReport::from(poly);
        report.tolc_order = tolc_order;
        self.receive_geometric_resonance(report);
    }

    pub fn last_report(&self) -> Option<&GeometricResonanceReport> {
        self.last_geometric_report.as_ref()
    }
}

impl Default for FractalMercyLedgerAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RaThorSystemAdapter for FractalMercyLedgerAdapter {
    fn system_name(&self) -> &'static str {
        self.name
    }

    fn current_valence(&self) -> Valence {
        self.current_valence
    }

    fn receive_swarm_resonance(&mut self, resonance: SwarmResonance) -> Result<(), MercyError> {
        // Swarm resonance can be mapped into a minimal geometric report
        // when a full PolyhedralResonanceReport is not available.
        println!(
            "[FractalMercyLedger] Received swarm resonance from {}: {:.3} — {}",
            resonance.source, resonance.intensity, resonance.message
        );

        if resonance.intensity > 0.75 {
            let boosted = (self.current_valence.value() + resonance.intensity * 0.000000008)
                .clamp(0.999999, 1.0);
            self.current_valence = Valence(boosted);
        }
        Ok(())
    }

    fn contribute_to_coherence(&self) -> GodlyIntelligenceCoherence {
        GodlyIntelligenceCoherence {
            precision: 0.95,
            resilience: 0.93,
            flow_stability: 0.91,
            harmonic_alignment: self.coherence_contribution,
        }
    }

    fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing) {
        self.blessing_count = self.blessing_count.saturating_add(1);
        println!(
            "[FractalMercyLedger] Applied epigenetic blessing: {} (strength {:.3}) [#{]]",
            blessing.blessing_type, blessing.strength, self.blessing_count
        );

        let mercy_boost = blessing.mercy_impact * 0.000000015;
        let new_val = (self.current_valence.value() + mercy_boost).clamp(0.999999, 1.0);
        self.current_valence = Valence(new_val);

        self.coherence_contribution =
            (self.coherence_contribution + blessing.evolution_impact * 0.008).clamp(0.88, 1.0);
    }

    fn status(&self) -> String {
        format!(
            "{}: valence={:.8} | coherence={:.4} | blessings={} | last_report={}",
            self.system_name(),
            self.current_valence.value(),
            self.coherence_contribution,
            self.blessing_count,
            self.last_geometric_report.is_some()
        )
    }
}

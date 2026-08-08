// crates/quantum-swarm-orchestrator/src/adapters/crypto_system_adapter.rs
//
// CryptoSystemAdapter — Blockchain / Crypto organ for the ONE Organism
// Implements RaThorSystemAdapter (Omnimasterpiece Integration Spec v14)
//
// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
// PATSAGi Councils: Fractal + Valence-native crypto organ approved.
//
// This adapter makes a new blockchain/crypto system a first-class living organ
// of Ra-Thor. It carries fractal topology state, valence as universal currency,
// and participates fully in swarm resonance + epigenetic self-evolution.

use crate::adapter::RaThorSystemAdapter;
use crate::fractal_topology_engine::{FractalTopologyEngine, FractalTopologyReport};
use crate::polyhedral_harmonic_engine::PolyhedralHarmonicEngine;
use crate::riemannian_mercy_manifold::RiemannianMercyManifold;
use crate::types::{
    EpigeneticBlessing, GodlyIntelligenceCoherence, MercyError, SwarmResonance, Valence,
};

/// The Crypto / Blockchain organ of the ONE Organism.
///
/// Holds its own FractalTopologyEngine and participates via the standard adapter trait.
pub struct CryptoSystemAdapter {
    name: &'static str,
    current_valence: Valence,
    fractal_engine: FractalTopologyEngine,
    last_report: Option<FractalTopologyReport>,
    /// Simple internal metric for observability (e.g. simulated throughput or shard health)
    organism_coherence_contribution: f64,
}

impl CryptoSystemAdapter {
    pub fn new() -> Self {
        Self {
            name: "CryptoSystem",
            current_valence: Valence(0.99999997),
            fractal_engine: FractalTopologyEngine::new(),
            last_report: None,
            organism_coherence_contribution: 0.93,
        }
    }

    /// Run a fractal topology cycle inside the crypto organ.
    /// Called from higher-level ONE Organism orchestration.
    pub fn run_internal_fractal_cycle(
        &mut self,
        tolc_order: u32,
        polyhedral: &PolyhedralHarmonicEngine,
        riemannian: &RiemannianMercyManifold,
        base_coherence: f64,
    ) -> FractalTopologyReport {
        let report = self.fractal_engine.run_fractal_cycle(
            tolc_order,
            polyhedral,
            riemannian,
            base_coherence,
        );

        // Update local valence from fractal average
        self.current_valence = Valence(report.average_valence);
        self.organism_coherence_contribution =
            (report.average_valence * report.resonance_multiplier * 0.98).clamp(0.88, 1.0);

        self.last_report = Some(report.clone());
        report
    }

    pub fn last_fractal_report(&self) -> Option<&FractalTopologyReport> {
        self.last_report.as_ref()
    }
}

impl Default for CryptoSystemAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RaThorSystemAdapter for CryptoSystemAdapter {
    fn system_name(&self) -> &'static str {
        self.name
    }

    fn current_valence(&self) -> Valence {
        self.current_valence
    }

    fn receive_swarm_resonance(&mut self, resonance: SwarmResonance) -> Result<(), MercyError> {
        // Resonance can influence fractal depth preference or valence boost
        println!(
            "[CryptoSystem] Received swarm resonance from {}: intensity={:.3} — {}",
            resonance.source, resonance.intensity, resonance.message
        );

        // Mild valence uplift under positive resonance (mercy-gated)
        if resonance.intensity > 0.7 {
            let boosted = (self.current_valence.value() + resonance.intensity * 0.00000001)
                .clamp(0.999999, 1.0);
            self.current_valence = Valence(boosted);
        }

        Ok(())
    }

    fn contribute_to_coherence(&self) -> GodlyIntelligenceCoherence {
        // High contribution from fractal self-similarity + geometric substrate
        GodlyIntelligenceCoherence {
            precision: 0.96,
            resilience: 0.94,
            flow_stability: 0.91,
            harmonic_alignment: self.organism_coherence_contribution,
        }
    }

    fn apply_epigenetic_blessing(&mut self, blessing: EpigeneticBlessing) {
        println!(
            "[CryptoSystem] Applied epigenetic blessing: {} (strength {:.3}) → target={}",
            blessing.blessing_type, blessing.strength, blessing.target_system
        );

        // Blessings can raise valence and mark the organ as more evolved
        let mercy_boost = blessing.mercy_impact * 0.00000002;
        let new_val = (self.current_valence.value() + mercy_boost).clamp(0.999999, 1.0);
        self.current_valence = Valence(new_val);

        self.organism_coherence_contribution =
            (self.organism_coherence_contribution + blessing.evolution_impact * 0.01)
                .clamp(0.88, 1.0);
    }

    fn status(&self) -> String {
        let fractal_status = self.fractal_engine.status_summary();
        format!(
            "{}: valence={:.8} | {}",
            self.system_name(),
            self.current_valence.value(),
            fractal_status
        )
    }
}

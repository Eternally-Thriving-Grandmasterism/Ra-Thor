//! lattice_conductor_v14.rs
//! Lattice Conductor v14 — ONE Organism aligned with full Geometric Intelligence Layer
//! Includes PolyhedralHarmonicEngine + ready for RiemannianMercyManifold scoring

use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use geometric_intelligence::{PolyhedralHarmonicEngine, PolyhedralResonanceReport};

use crate::attom::AttomData;
use crate::mercy_gates::MercyGate;

#[derive(Debug)]
pub struct LatticeConductor {
    pub version: &'static str,
    pub mercy_gates: Vec<MercyGate>,
    pub attom_cache: DashMap<String, Arc<AttomData>>,
    pub regulatory_rules: HashMap<String, String>,
    pub geometric_engine: PolyhedralHarmonicEngine,
}

impl LatticeConductor {
    pub fn new() -> Self {
        let gates = vec![
            MercyGate::Truth,
            MercyGate::Order,
            MercyGate::Love,
            MercyGate::Compassion,
            MercyGate::Service,
            MercyGate::Abundance,
            MercyGate::Joy,
            MercyGate::CosmicHarmony,
        ];

        let mut rules = HashMap::new();
        rules.insert(
            "Ontario".to_string(),
            "RESA/TRESA compliance + reverse onus safety checks".to_string(),
        );
        rules.insert(
            "USA".to_string(),
            "State-level disclosure + federal fair housing".to_string(),
        );

        let attom_cache: DashMap<String, Arc<AttomData>> = DashMap::with_shard_amount(32);

        LatticeConductor {
            version: "v14.4.0-geometric-intelligence",
            mercy_gates: gates,
            attom_cache,
            regulatory_rules: rules,
            geometric_engine: PolyhedralHarmonicEngine::new(),
        }
    }

    /// Computes polyhedral resonance-based geometric harmony for a real estate offer.
    pub fn compute_geometric_harmony_score(
        &self,
        jurisdiction: &str,
        tolc_order: u32,
        base_coherence: f64,
    ) -> f64 {
        let report = self
            .geometric_engine
            .compute_full_resonance_report(tolc_order, base_coherence);

        let jurisdiction_bonus = if jurisdiction == "Ontario" || jurisdiction == "USA" {
            1.03
        } else {
            1.0
        };

        let score = report.resonance_multiplier * jurisdiction_bonus * base_coherence;
        score.clamp(0.6, 1.65)
    }

    /// Returns a full resonance report for deeper analysis.
    pub fn get_geometric_resonance_report(
        &self,
        tolc_order: u32,
        base_coherence: f64,
    ) -> PolyhedralResonanceReport {
        self.geometric_engine
            .compute_full_resonance_report(tolc_order, base_coherence)
    }

    /// Extension point for future RiemannianMercyManifold integration.
    pub fn compute_riemannian_adjusted_harmony(
        &self,
        base_score: f64,
        _curvature_influence: f64,
    ) -> f64 {
        base_score
    }
}

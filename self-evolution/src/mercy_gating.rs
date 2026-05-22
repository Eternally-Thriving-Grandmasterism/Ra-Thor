//! Coordinated Deep Implementation: MaatKpi Centrality + Cross-Layer + Priority Gates

// MaatKpi is now elevated to serve as a shared scoring mechanism across layers.

#[derive(Debug, Clone)]
pub struct MaatKpi {
    pub dimension_scores: HashMap<MaatDimension, f64>,
}

impl MaatKpi {
    pub fn new() -> Self { Self::default() }

    pub fn set_score(&mut self, dimension: MaatDimension, score: f64) {
        self.dimension_scores.insert(dimension, score.clamp(0.0, 1.0));
    }

    pub fn overall_score(&self) -> f64 {
        if self.dimension_scores.is_empty() { return 0.0; }
        self.dimension_scores.values().sum::<f64>() / self.dimension_scores.len() as f64
    }

    /// New: Can now be used to influence multiple layers
    pub fn layer_adjusted_score(&self, layer: MercyGateLevel) -> f64 {
        let base = self.overall_score();
        match layer {
            MercyGateLevel::Foundational => base * 0.95,
            MercyGateLevel::Operational => base,
            MercyGateLevel::Integrative => base * 1.05, // Slightly higher weight at integrative level
        }
    }
}

// ... existing code continues with updated evaluation using layer_adjusted_score ...

// Initial deeper implementation for LatticeCoherence and PatsagiConsensus
fn evaluate_lattice_coherence(base_score: f64, kpi: &MaatKpi) -> MercyVerdict {
    let adjusted = kpi.layer_adjusted_score(MercyGateLevel::Integrative);
    if adjusted >= 0.89 {
        MercyVerdict::Mitigated {
            overall_score: adjusted,
            notes: vec!["Lattice Coherence: Strong structural + Ma'at alignment".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

fn evaluate_patsagi_consensus(base_score: f64, kpi: &MaatKpi) -> MercyVerdict {
    let adjusted = kpi.layer_adjusted_score(MercyGateLevel::Integrative);
    if adjusted >= 0.90 {
        MercyVerdict::Mitigated {
            overall_score: adjusted,
            notes: vec!["PATSAGi Consensus: High multi-council + Ma'at coherence".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// ... rest of file ...
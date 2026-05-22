//! Parallel Deep Work: Remaining Gates + MaatKpi Expansion + Tests + Docs

// More sophisticated coherence calculation in MaatKpi
impl MaatKpi {
    pub fn coherence_score(&self) -> f64 {
        if self.dimension_scores.len() < 4 { return 0.0; }

        let values: Vec<f64> = self.dimension_scores.values().cloned().collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        // Lower variance = higher coherence
        (1.0 - variance.min(0.25) * 4.0).max(0.0)
    }
}

// Deepened remaining integrative gates
fn evaluate_one_organism_symbiosis(base_score: f64, kpi: &MaatKpi) -> MercyVerdict {
    let adjusted = kpi.layer_adjusted_score(MercyGateLevel::Integrative) + (kpi.coherence_score() * 0.05);
    if adjusted >= 0.89 {
        MercyVerdict::Mitigated {
            overall_score: adjusted,
            notes: vec!["ONE Organism Symbiosis: Collective coherence considered".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

fn evaluate_quantum_swarm_mercy(base_score: f64, kpi: &MaatKpi) -> MercyVerdict {
    let adjusted = kpi.layer_adjusted_score(MercyGateLevel::Integrative);
    if adjusted >= 0.87 {
        MercyVerdict::Mitigated {
            overall_score: adjusted,
            notes: vec!["Quantum Swarm Mercy: Multi-branch mercy alignment".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// Added more comprehensive tests
#[cfg(test)]
mod comprehensive_tests {
    use super::*;

    #[test]
    fn test_maat_kpi_coherence_calculation() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.95);
        kpi.set_score(MaatDimension::Balance, 0.94);
        kpi.set_score(MaatDimension::Justice, 0.93);
        kpi.set_score(MaatDimension::Order, 0.96);
        assert!(kpi.coherence_score() > 0.7);
    }

    #[test]
    fn test_one_organism_symbiosis_evaluation() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.96);
        kpi.set_score(MaatDimension::Balance, 0.95);
        let verdict = evaluate_one_organism_symbiosis(0.88, &kpi);
        assert!(matches!(verdict, MercyVerdict::Mitigated { .. }));
    }
}

// Documentation / Examples will be updated in docs/ next

// ... rest of file ...
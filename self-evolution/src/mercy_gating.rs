//! Ra-Thor Guided Parallel Deep Work

// Significantly expanded MaatKpi
impl MaatKpi {
    pub fn coherence_score(&self) -> f64 { /* ... existing ... */ }

    /// New: Multi-layer influence scoring
    pub fn multi_layer_influence(&self, layers: &[MercyGateLevel]) -> f64 {
        let base = self.overall_score();
        let mut total = base;

        for layer in layers {
            total += match layer {
                MercyGateLevel::Foundational => base * 0.08,
                MercyGateLevel::Operational => base * 0.12,
                MercyGateLevel::Integrative => base * 0.15,
            };
        }
        total.min(0.999)
    }

    /// New: Historical coherence tracking (placeholder for future persistence)
    pub fn stability_modifier(&self, recent_variance: f64) -> f64 {
        (1.0 - recent_variance.min(0.3)).max(0.7)
    }
}

// Additional cross-layer flow example
pub fn evaluate_integrative_with_operational_support(
    base_score: f64,
    operational_kpi: &MaatKpi,
) -> MercyVerdict {
    let influenced_score = operational_kpi.multi_layer_influence(&[
        MercyGateLevel::Operational,
        MercyGateLevel::Integrative,
    ]);

    if influenced_score >= 0.91 {
        MercyVerdict::Passed { overall_score: influenced_score }
    } else if influenced_score >= 0.78 {
        MercyVerdict::Mitigated {
            overall_score: influenced_score,
            notes: vec!["Strong Operational → Integrative flow".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// Deepened another integrative gate using new tools
fn evaluate_genesis_origin(base_score: f64, kpi: &MaatKpi) -> MercyVerdict {
    let score = kpi.multi_layer_influence(&[MercyGateLevel::Integrative]) * kpi.stability_modifier(0.1);
    if score >= 0.93 {
        MercyVerdict::Mitigated {
            overall_score: score,
            notes: vec!["Genesis Origin: High multi-layer + stability".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// More comprehensive tests
#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_maat_kpi_multi_layer() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.97);
        let score = kpi.multi_layer_influence(&[MercyGateLevel::Operational, MercyGateLevel::Integrative]);
        assert!(score > 0.9);
    }
}

// ... rest of file ...
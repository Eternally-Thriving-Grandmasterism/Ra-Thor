//! Cross-Layer Interaction Logic + Production Examples

/// Cross-layer evaluation helper
/// Allows Foundational and Operational results to influence Integrative decisions
pub fn evaluate_with_cross_layer(
    base_score: f64,
    foundational_verdict: Option<MercyVerdict>,
    operational_kpi: Option<&MaatKpi>,
    target_layer: MercyGateLevel,
) -> MercyVerdict {
    let mut adjusted_score = base_score;

    // Foundational influence
    if let Some(MercyVerdict::Mitigated { overall_score, .. }) = foundational_verdict {
        adjusted_score = (adjusted_score + overall_score * 0.15).min(0.999);
    }

    // Operational (Ma'at) influence
    if let Some(kpi) = operational_kpi {
        let maat_influence = kpi.layer_adjusted_score(MercyGateLevel::Operational) * 0.2;
        adjusted_score = (adjusted_score + maat_influence).min(0.999);
    }

    // Final decision at target layer
    if adjusted_score >= 0.90 {
        MercyVerdict::Passed { overall_score: adjusted_score }
    } else if adjusted_score >= 0.75 {
        MercyVerdict::Mitigated {
            overall_score: adjusted_score,
            notes: vec!["Cross-layer synergy applied".to_string()],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// ... existing code ...

#[cfg(test)]
mod cross_layer_tests {
    use super::*;

    #[test]
    fn test_cross_layer_influence() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.96);
        kpi.set_score(MaatDimension::Balance, 0.94);

        let verdict = evaluate_with_cross_layer(0.82, None, Some(&kpi), MercyGateLevel::Integrative);
        assert!(verdict_overall_score(&verdict) > 0.82);
    }
}

fn verdict_overall_score(verdict: &MercyVerdict) -> f64 {
    match verdict {
        MercyVerdict::Passed { overall_score } => *overall_score,
        MercyVerdict::Mitigated { overall_score, .. } => *overall_score,
        _ => 0.0,
    }
}

// ... rest of file ...
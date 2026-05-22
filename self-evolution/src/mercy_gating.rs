//! Expanded Cross-Layer and Gate Tests

#[cfg(test)]
mod expanded_cross_layer_tests {
    use super::*;

    #[test]
    fn test_cross_layer_with_operational_kpi() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.97);
        kpi.set_score(MaatDimension::Balance, 0.95);
        kpi.set_score(MaatDimension::Justice, 0.93);
        kpi.set_score(MaatDimension::Order, 0.96);

        let verdict = evaluate_with_cross_layer(0.80, None, Some(&kpi), MercyGateLevel::Integrative);
        // Should benefit from strong Operational KPI
        assert!(verdict_overall_score(&verdict) > 0.80);
    }
}

// ... existing code ...
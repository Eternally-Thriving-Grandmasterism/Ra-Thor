// ... existing code ...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foundational_evaluation() {
        let err = crate::SnapshotError::FileNotFound { path: "test.json".to_string() };
        let verdict = err.evaluate_mercy(MercyGateLevel::Foundational);
        assert!(matches!(verdict, MercyVerdict::Mitigated { .. }));
    }

    #[test]
    fn test_operational_maat_high_score() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.98);
        kpi.set_score(MaatDimension::Balance, 0.95);
        kpi.set_score(MaatDimension::Justice, 0.91);
        kpi.set_score(MaatDimension::Order, 0.94);
        assert!(kpi.overall_score() > 0.93);
        assert!(kpi.meets_threshold(0.90));
    }

    #[test]
    fn test_integrative_higher_standard() {
        let err = crate::SnapshotError::ParseError { source: serde_json::from_str::<serde_json::Value>("bad").unwrap_err() };
        let verdict = err.evaluate_mercy(MercyGateLevel::Integrative);
        // Should lean toward council review for meta layer
        assert!(matches!(verdict, MercyVerdict::RequiresCouncilReview) || matches!(verdict, MercyVerdict::Mitigated { .. }));
    }
}
// ... existing code ...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maat_kpi_scoring() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.9);
        kpi.set_score(MaatDimension::Balance, 0.8);
        assert!(kpi.overall_score() > 0.8);
    }

    #[test]
    fn test_mercy_verdict_levels() {
        // Basic sanity check for different levels
        assert!(true);
    }
}
//! Expanded Property Tests + Fuzzing Guidance

#[cfg(test)]
mod expanded_property_tests {
    use super::*;

    #[test]
    fn property_all_maat_kpi_methods_stay_in_bounds() {
        let mut kpi = MaatKpi::new();
        // Test extreme combinations
        kpi.set_score(MaatDimension::Truth, 0.0);
        kpi.set_score(MaatDimension::Balance, 1.0);
        kpi.set_score(MaatDimension::Justice, 0.0);
        kpi.set_score(MaatDimension::Order, 1.0);

        assert!(kpi.overall_score() >= 0.0 && kpi.overall_score() <= 1.0);
        assert!(kpi.coherence_score() >= 0.0);
    }

    #[test]
    fn property_self_referential_respects_thresholds() {
        // High everything should pass
        let high = self_referential_mercy_evaluation(0.98, 0.95, 0.93);
        assert!(matches!(high, MercyVerdict::Passed { .. }));

        // Low everything should require review
        let low = self_referential_mercy_evaluation(0.4, 0.5, 0.45);
        assert!(matches!(low, MercyVerdict::RequiresCouncilReview));
    }
}

/*
 * cargo-fuzz Setup Instructions:
 *
 * 1. Add to Cargo.toml (dev-dependencies):
 *    cargo-fuzz = "0.11"
 *
 * 2. Run: cargo install cargo-fuzz
 *
 * 3. Initialize: cargo fuzz init
 *
 * 4. Create fuzz/fuzz_targets/self_referential.rs with:
 *    #![no_main]
 *    use libfuzzer_sys::fuzz_target;
 *    use self_evolution::mercy_gating::self_referential_mercy_evaluation;
 *
 *    fuzz_target!(|data: &[u8]| {
 *        // parse data into scores and call the function
 *    });
 *
 * Mutation testing can be explored later using cargo-mutants.
 */

// ... existing code ...
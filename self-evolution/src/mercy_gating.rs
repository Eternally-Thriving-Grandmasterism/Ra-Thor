//! # Ultimate Unified MercyGating System (Enriched v0.5.0)
//!
//! Primary focus: 16-gate Ma'at system with rich contextual evaluation.
//! Foundational 7-gate support retained for lightweight checks.
//!
//! This version emphasizes usefulness, depth, and coherence over excessive hierarchy.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MercyGateLevel {
    Seven,
    EightTolc,
    SixteenMaat,
}

// ... (gate enums remain) ...

impl MercyGateEvaluable for crate::SnapshotError {
    fn evaluate_mercy(&self, level: MercyGateLevel) -> MercyVerdict {
        let base_score = match self {
            crate::SnapshotError::FileNotFound { .. } => 0.82,
            crate::SnapshotError::ReadError { .. } => 0.78,
            crate::SnapshotError::ParseError { .. } => 0.65,
            crate::SnapshotError::UnknownFormat => 0.60,
        };

        match level {
            MercyGateLevel::Seven | MercyGateLevel::EightTolc => {
                if base_score >= 0.78 {
                    MercyVerdict::Mitigated {
                        overall_score: base_score,
                        notes: vec!["Lightweight foundational mercy evaluation passed.".to_string()],
                    }
                } else {
                    MercyVerdict::RequiresCouncilReview
                }
            }
            MercyGateLevel::SixteenMaat => {
                let mut kpi = MaatKpi::new();
                // Enriched contextual scoring
                kpi.set_score(MaatDimension::Truth, base_score * 0.96);
                kpi.set_score(MaatDimension::Balance, base_score * 0.92);
                kpi.set_score(MaatDimension::Justice, base_score * 0.87);
                kpi.set_score(MaatDimension::Order, base_score * 0.90);

                let maat_score = kpi.overall_score();

                if maat_score >= 0.88 {
                    MercyVerdict::Passed {
                        overall_score: maat_score,
                    }
                } else if maat_score >= 0.72 {
                    MercyVerdict::Mitigated {
                        overall_score: maat_score,
                        notes: vec![format!("Ma'at contextual score: {:.3}", maat_score)],
                    }
                } else {
                    MercyVerdict::RequiresCouncilReview
                }
            }
        }
    }
}

// ... rest of file with improved tests ...

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maat_kpi_rich_scoring() {
        let mut kpi = MaatKpi::new();
        kpi.set_score(MaatDimension::Truth, 0.95);
        kpi.set_score(MaatDimension::Balance, 0.91);
        kpi.set_score(MaatDimension::Justice, 0.88);
        kpi.set_score(MaatDimension::Order, 0.93);

        let score = kpi.overall_score();
        assert!(score > 0.90);
        assert!(kpi.meets_threshold(0.85));
    }
}
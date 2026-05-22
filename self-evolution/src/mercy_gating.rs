//! ... existing code ...

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
                if base_score >= 0.75 {
                    MercyVerdict::Mitigated {
                        overall_score: base_score,
                        notes: vec!["Evaluated through foundational Mercy + TOLC gates".to_string()],
                    }
                } else {
                    MercyVerdict::RequiresCouncilReview
                }
            }
            MercyGateLevel::SixteenMaat => {
                // Use Ma'at KPI scoring for higher granularity
                let mut kpi = MaatKpi::new();
                kpi.set_score(MaatDimension::Truth, base_score * 0.95);
                kpi.set_score(MaatDimension::Balance, base_score * 0.90);
                kpi.set_score(MaatDimension::Justice, base_score * 0.85);
                kpi.set_score(MaatDimension::Order, base_score * 0.88);

                let maat_score = kpi.overall_score();

                if maat_score >= 0.85 {
                    MercyVerdict::Passed { overall_score: maat_score }
                } else if maat_score >= 0.70 {
                    MercyVerdict::Mitigated {
                        overall_score: maat_score,
                        notes: vec![format!("Ma'at overall: {:.2}", maat_score)],
                    }
                } else {
                    MercyVerdict::RequiresCouncilReview
                }
            }
        }
    }
}
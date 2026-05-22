// ... existing code ...

impl mercy_gating::MercyGateEvaluable for SnapshotError {
    fn evaluate_mercy(&self, level: mercy_gating::MercyGateLevel) -> mercy_gating::MercyVerdict {
        let base_score = match self {
            SnapshotError::FileNotFound { .. } => 0.82,
            SnapshotError::ReadError { .. } => 0.78,
            SnapshotError::ParseError { .. } => 0.65,
            SnapshotError::UnknownFormat => 0.60,
        };

        match level {
            mercy_gating::MercyGateLevel::Seven | mercy_gating::MercyGateLevel::EightTolc => {
                if base_score >= 0.75 {
                    mercy_gating::MercyVerdict::Mitigated {
                        overall_score: base_score,
                        notes: vec!["Evaluated through foundational Mercy Gates".to_string()],
                    }
                } else {
                    mercy_gating::MercyVerdict::RequiresCouncilReview
                }
            }
            mercy_gating::MercyGateLevel::SixteenMaat => {
                // Phase 1: Proper Ma'at KPI scoring
                let mut kpi = mercy_gating::MaatKpi::new();
                kpi.set_score(mercy_gating::MaatDimension::Truth, base_score * 0.95);
                kpi.set_score(mercy_gating::MaatDimension::Balance, base_score * 0.90);
                kpi.set_score(mercy_gating::MaatDimension::Justice, base_score * 0.85);
                kpi.set_score(mercy_gating::MaatDimension::Order, base_score * 0.88);

                let maat_score = kpi.overall_score();

                if maat_score >= 0.85 {
                    mercy_gating::MercyVerdict::Passed { overall_score: maat_score }
                } else if maat_score >= 0.70 {
                    mercy_gating::MercyVerdict::Mitigated {
                        overall_score: maat_score,
                        notes: vec![format!("Ma'at score: {:.2}", maat_score)],
                    }
                } else {
                    mercy_gating::MercyVerdict::RequiresCouncilReview
                }
            }
        }
    }
}

// ... existing code ...
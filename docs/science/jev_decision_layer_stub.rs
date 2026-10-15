//! Decision-layer documentation stub (research only).
//!
//! Workspace 14.15.6. Contact: info@Rathor.ai
//! License: AG-SML v1.1. Human review required.
//!
//! Public claim: inspectable research software; optional Grok session;
//! not xAI affiliated; not certified; not a legal product.
//!
//! signal only, not a gate
//!
//! This file is a local shape for a typed Choice, an ordered Score, and a
//! noul-style probability that a later model call might sit behind.
//! It does not call TypeSafe, OpenRouter, or any external service.
//! It does not load sampler weights. It does not admit, reject, or apply.
//! A reading here cannot flip a Layer 0 Reject into an Apply.
//!
//! No concrete high-volume use case is named. The read-first note
//! `docs/science/PATSAGI-NOTE-2026-10-15-JEV-DECISION-LAYER.md` is absent
//! from this tree. Work stops at this stub.
//!
//! Pure-local fallback: [`local_abstain_noul`] returns P(true) = 1/2.
//! That value is an abstain prior. It is not a measurement.
//! Low-confidence readings escalate to a larger model the operator already
//! runs (open-weight or other pure-local host) or to a human. This file
//! does not perform that call.
//!
//! Compile outside the workspace members:
//! `rustc --edition 2021 --test docs/science/jev_decision_layer_stub.rs`

/// Exact comment the task requires on this stub.
pub const SIGNAL_ONLY_NOT_A_GATE: &str = "signal only, not a gate";

/// Public claim line. Workspace identity stays 14.15.6.
pub const PUBLIC_CLAIM_LINE: &str =
    "inspectable research software; optional Grok session; not xAI affiliated; not certified; not a legal product.";

/// One label in a Choice question.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedLabel {
    /// Operator-supplied label. This stub does not invent categories.
    pub id: &'static str,
    /// Signal in \[0, 1\]. Not a calibrated frequency and not a gate.
    pub probability: f64,
}

/// Typed choice: one selected label plus a probability for each option.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceSignal {
    /// Label the probabilities mark as the mode.
    pub chosen: &'static str,
    /// Full distribution over the supplied options.
    pub probabilities: Vec<WeightedLabel>,
}

/// Ordered score: a level index plus a probability for each rubric level.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreSignal {
    /// Index into the caller-supplied ordered rubric.
    pub level_index: usize,
    /// Probability for each level, in rubric order.
    pub probabilities: Vec<WeightedLabel>,
}

/// Noul-style probability: estimated P(proposition is true), in \[0, 1\].
///
/// This is one number for one yes/no proposition. It is not a second
/// confidence field, and it is not proof that the proposition holds.
#[derive(Clone, Debug, PartialEq)]
pub struct NoulSignal {
    /// Estimated P(true).
    pub probability_true: f64,
}

/// Signals that would sit in front of a later model call.
///
/// Every field is optional. Absence is an incomplete record.
#[derive(Clone, Debug, PartialEq)]
pub struct FrontSignal {
    /// Typed choice, when the caller has one.
    pub choice: Option<ChoiceSignal>,
    /// Ordered score, when the caller has one.
    pub score: Option<ScoreSignal>,
    /// Noul-style probability, when the caller has one.
    pub noul: Option<NoulSignal>,
}

/// Caller-supplied bands. This stub does not ship a measured threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadPolicy {
    /// Noul values within this distance of 1/2 escalate.
    pub noul_escalate_band: f64,
    /// Choice or score mode must reach at least this probability to record.
    pub min_top: f64,
    /// Gap between the top two probabilities required to record.
    pub min_margin: f64,
}

/// What a human may do next. Constructing a variant does not perform it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalReading {
    /// The supplied numbers sit outside the caller’s low-confidence band.
    /// Recording them is not permission to apply, admit, or call a service.
    Recorded,
    /// Low confidence, a malformed record, or no signal at all.
    /// Escalate to a larger model the operator already runs, or to a human.
    /// This arm does not perform that escalation.
    EscalateToLargerModelOrHuman,
}

/// Readings for each present signal, plus one combined escalate flag.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontReading {
    /// Choice reading, present only when a choice was supplied.
    pub choice: Option<SignalReading>,
    /// Score reading, present only when a score was supplied.
    pub score: Option<SignalReading>,
    /// Noul reading, present only when a noul was supplied.
    pub noul: Option<SignalReading>,
    /// True when any present signal escalates, or when none were supplied.
    pub escalate: bool,
}

/// Pure-local fallback when no open-weight decider is running.
///
/// Returns P(true) = 1/2. That is an abstain prior, not a measurement,
/// not a calibrated probability, and not a gate.
pub fn local_abstain_noul() -> NoulSignal {
    NoulSignal {
        probability_true: 0.5,
    }
}

/// Read one noul-style probability under a caller-supplied band.
///
/// signal only, not a gate
///
/// Low confidence (within `escalate_band` of 1/2), a non-finite value, or a
/// value outside \[0, 1\] escalates to a larger model the operator already
/// runs, or to a human. This function does not call TypeSafe, OpenRouter,
/// or any external service.
pub fn read_noul(probability_true: f64, escalate_band: f64) -> SignalReading {
    if !probability_true.is_finite()
        || !escalate_band.is_finite()
        || escalate_band < 0.0
        || !(0.0..=1.0).contains(&probability_true)
    {
        return SignalReading::EscalateToLargerModelOrHuman;
    }
    let distance_from_half = (probability_true - 0.5).abs();
    if distance_from_half <= escalate_band {
        // Low confidence: escalate to a larger model the operator already
        // runs (prefer an open-weight or other pure-local host) or to a human.
        // Do not call TypeSafe, OpenRouter, or any external service from here.
        return SignalReading::EscalateToLargerModelOrHuman;
    }
    SignalReading::Recorded
}

/// Read a typed choice. The chosen id must be the unique mode.
///
/// A flat distribution, a top below `min_top`, or a margin below `min_margin`
/// escalates. This function does not call a model.
pub fn read_choice(signal: &ChoiceSignal, min_top: f64, min_margin: f64) -> SignalReading {
    match mode_of(&signal.probabilities, min_top, min_margin) {
        Some(mode_id) if mode_id == signal.chosen => SignalReading::Recorded,
        _ => SignalReading::EscalateToLargerModelOrHuman,
    }
}

/// Read an ordered score. `level_index` must name the mode’s position.
///
/// A flat distribution escalates. This function does not call a model.
pub fn read_score(signal: &ScoreSignal, min_top: f64, min_margin: f64) -> SignalReading {
    match mode_index(&signal.probabilities, min_top, min_margin) {
        Some(index) if index == signal.level_index => SignalReading::Recorded,
        _ => SignalReading::EscalateToLargerModelOrHuman,
    }
}

/// Read a bundle that would sit in front of a later model call.
///
/// An empty bundle escalates. Nothing in this function places the call.
/// signal only, not a gate
pub fn read_front(front: &FrontSignal, policy: &ReadPolicy) -> FrontReading {
    let choice = front
        .choice
        .as_ref()
        .map(|signal| read_choice(signal, policy.min_top, policy.min_margin));
    let score = front
        .score
        .as_ref()
        .map(|signal| read_score(signal, policy.min_top, policy.min_margin));
    let noul = front
        .noul
        .as_ref()
        .map(|signal| read_noul(signal.probability_true, policy.noul_escalate_band));
    let escalate = match (&choice, &score, &noul) {
        (None, None, None) => true,
        _ => [&choice, &score, &noul]
            .into_iter()
            .flatten()
            .any(|reading| *reading == SignalReading::EscalateToLargerModelOrHuman),
    };
    FrontReading {
        choice,
        score,
        noul,
        escalate,
    }
}

fn mode_of<'a>(rows: &'a [WeightedLabel], min_top: f64, min_margin: f64) -> Option<&'a str> {
    mode_index(rows, min_top, min_margin).map(|index| rows[index].id)
}

fn mode_index(rows: &[WeightedLabel], min_top: f64, min_margin: f64) -> Option<usize> {
    if rows.is_empty() || !min_top.is_finite() || !min_margin.is_finite() || min_margin < 0.0 {
        return None;
    }
    let mut sum = 0.0;
    let mut best_i = 0;
    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for (i, row) in rows.iter().enumerate() {
        let p = row.probability;
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return None;
        }
        sum += p;
        if p > best {
            second = best;
            best = p;
            best_i = i;
        } else if p > second {
            second = p;
        }
    }
    if (sum - 1.0).abs() > 1e-6 || best < min_top {
        return None;
    }
    if second.is_finite() && (best - second) < min_margin {
        return None;
    }
    Some(best_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ReadPolicy {
        ReadPolicy {
            noul_escalate_band: 0.1,
            min_top: 0.6,
            min_margin: 0.2,
        }
    }

    #[test]
    fn claim_and_signal_lines_are_present() {
        assert_eq!(SIGNAL_ONLY_NOT_A_GATE, "signal only, not a gate");
        assert!(PUBLIC_CLAIM_LINE.contains("inspectable research software"));
        assert!(PUBLIC_CLAIM_LINE.contains("optional Grok session"));
        assert!(PUBLIC_CLAIM_LINE.contains("not xAI affiliated"));
        assert!(PUBLIC_CLAIM_LINE.contains("not certified"));
        assert!(PUBLIC_CLAIM_LINE.contains("not a legal product"));
    }

    #[test]
    fn pure_local_abstain_escalates() {
        let prior = local_abstain_noul();
        assert_eq!(prior.probability_true, 0.5);
        assert_eq!(
            read_noul(prior.probability_true, 0.0),
            SignalReading::EscalateToLargerModelOrHuman
        );
    }

    #[test]
    fn decisive_noul_records_and_near_half_escalates() {
        assert_eq!(read_noul(0.91, 0.1), SignalReading::Recorded);
        assert_eq!(
            read_noul(0.6, 0.1),
            SignalReading::EscalateToLargerModelOrHuman
        );
        assert_eq!(
            read_noul(f64::NAN, 0.1),
            SignalReading::EscalateToLargerModelOrHuman
        );
        assert_eq!(
            read_noul(1.2, 0.1),
            SignalReading::EscalateToLargerModelOrHuman
        );
    }

    #[test]
    fn peaked_choice_records_and_flat_choice_escalates() {
        let peaked = ChoiceSignal {
            chosen: "hold",
            probabilities: vec![
                WeightedLabel {
                    id: "hold",
                    probability: 0.8,
                },
                WeightedLabel {
                    id: "review",
                    probability: 0.2,
                },
            ],
        };
        assert_eq!(read_choice(&peaked, 0.6, 0.2), SignalReading::Recorded);

        let flat = ChoiceSignal {
            chosen: "hold",
            probabilities: vec![
                WeightedLabel {
                    id: "hold",
                    probability: 0.51,
                },
                WeightedLabel {
                    id: "review",
                    probability: 0.49,
                },
            ],
        };
        assert_eq!(
            read_choice(&flat, 0.6, 0.2),
            SignalReading::EscalateToLargerModelOrHuman
        );
    }

    #[test]
    fn recorded_front_stays_a_signal() {
        let reading = read_front(
            &FrontSignal {
                choice: Some(ChoiceSignal {
                    chosen: "hold",
                    probabilities: vec![
                        WeightedLabel {
                            id: "hold",
                            probability: 0.8,
                        },
                        WeightedLabel {
                            id: "review",
                            probability: 0.2,
                        },
                    ],
                }),
                score: Some(ScoreSignal {
                    level_index: 1,
                    probabilities: vec![
                        WeightedLabel {
                            id: "low",
                            probability: 0.15,
                        },
                        WeightedLabel {
                            id: "high",
                            probability: 0.85,
                        },
                    ],
                }),
                noul: Some(NoulSignal {
                    probability_true: 0.91,
                }),
            },
            &policy(),
        );
        assert_eq!(reading.choice, Some(SignalReading::Recorded));
        assert_eq!(reading.score, Some(SignalReading::Recorded));
        assert_eq!(reading.noul, Some(SignalReading::Recorded));
        assert!(!reading.escalate);
    }

    #[test]
    fn empty_front_escalates_without_a_model_call() {
        let reading = read_front(
            &FrontSignal {
                choice: None,
                score: None,
                noul: None,
            },
            &policy(),
        );
        assert!(reading.escalate);
        assert_eq!(reading.choice, None);
        assert_eq!(reading.noul, None);
    }

    #[test]
    fn score_must_name_the_mode() {
        let aligned = ScoreSignal {
            level_index: 1,
            probabilities: vec![
                WeightedLabel {
                    id: "low",
                    probability: 0.15,
                },
                WeightedLabel {
                    id: "high",
                    probability: 0.85,
                },
            ],
        };
        assert_eq!(read_score(&aligned, 0.6, 0.2), SignalReading::Recorded);

        let misindexed = ScoreSignal {
            level_index: 0,
            probabilities: aligned.probabilities.clone(),
        };
        assert_eq!(
            read_score(&misindexed, 0.6, 0.2),
            SignalReading::EscalateToLargerModelOrHuman
        );
    }
}

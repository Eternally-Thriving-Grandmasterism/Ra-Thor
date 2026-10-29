//! `apply_council_tuning` refuses a lower `new_value` and keeps the stored value.
//! A proposal of `100.0` leaves `ma_at_threshold` at `717.0`.
//! A proposal of `755.0` stores `755.0`. A named proposal of `0.92` stores `0.92`.

use mercy_gating_runtime::{CouncilTuningProposal, MercyGatingRuntime, TuningTarget};

#[test]
fn test_phase4_lower_proposal_is_rejected_and_717_stays() {
    let mut runtime = MercyGatingRuntime::default();
    let malicious_low = CouncilTuningProposal {
        council_id: 99,
        target: TuningTarget::MaAtThreshold,
        new_value: 100.0, // attempt to lower
        justification: "Attempt to lower standard".to_string(),
        proposed_at_turn: 1,
    };
    let result = runtime.apply_council_tuning(&malicious_low);
    assert!(!result.success);
    assert_eq!(result.previous_value, 717.0);
    assert_eq!(result.new_value, 717.0);
    assert_eq!(runtime.ma_at_threshold, 717.0);
}

#[test]
fn test_phase4_raise_to_755_stores_755() {
    let mut runtime = MercyGatingRuntime::default();
    let initial = runtime.ma_at_threshold;

    let proposal = CouncilTuningProposal {
        council_id: 13,
        target: TuningTarget::MaAtThreshold,
        new_value: 755.0,
        justification: "Raise for higher coherence".to_string(),
        proposed_at_turn: 10,
    };
    let result = runtime.apply_council_tuning(&proposal);
    assert!(result.success);
    assert_eq!(initial, 717.0);
    assert_eq!(runtime.ma_at_threshold, 755.0);
}

#[test]
fn test_phase4_740_then_780_stores_780() {
    let mut runtime = MercyGatingRuntime::default();
    let initial = runtime.ma_at_threshold;

    let p1 = CouncilTuningProposal {
        council_id: 13,
        target: TuningTarget::MaAtThreshold,
        new_value: 740.0,
        justification: "First raise".to_string(),
        proposed_at_turn: 5,
    };
    let p2 = CouncilTuningProposal {
        council_id: 24,
        target: TuningTarget::MaAtThreshold,
        new_value: 780.0,
        justification: "Second raise during arbitration".to_string(),
        proposed_at_turn: 6,
    };

    let _ = runtime.apply_council_tunings(&[p1, p2]);
    assert_eq!(initial, 717.0);
    assert_eq!(runtime.ma_at_threshold, 780.0);
}

#[test]
fn test_phase4_gate_threshold_proposal_stores_0_92() {
    let mut runtime = MercyGatingRuntime::default();
    let proposals = vec![CouncilTuningProposal {
        council_id: 13,
        target: TuningTarget::GateThreshold {
            gate: "one_organism_unity".to_string(),
        },
        new_value: 0.92,
        justification: "Tighten unity gate".to_string(),
        proposed_at_turn: 20,
    }];
    let results = runtime.apply_council_tunings(&proposals);
    assert!(!results.is_empty());
    assert!(results[0].success);
    assert_eq!(results[0].new_value, 0.92);
    assert_eq!(runtime.get_gate_threshold("one_organism_unity"), 0.92);
}

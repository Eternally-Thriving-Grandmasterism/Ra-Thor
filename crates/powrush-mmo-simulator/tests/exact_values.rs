//! Exact values for the compiled simulator surface.
//! MWPO figures are the mial source-literal formulas.
//! Effect multipliers are the arithmetic in `gate_logic`.

use mial::mwpo::{GeometryParams, MercyContext, MercyGeometryScore, SacredSolid, SymmetryGroup};
use powrush_mmo_simulator::{
    abundance_system_reaction, apply_cooperation_bonus, apply_evolution_stability,
    apply_formal_confirmation_bonus, apply_formal_verification_failure_penalty,
    apply_geometry_structural_bonus, apply_information_accuracy, apply_resource_generation,
    composite_gate_health, compute_gate_effects, emit_and_collect_gate_events,
    evaluate_particle_geometry_mercy, evolution_system_reaction, faction_harmony_reaction,
    formally_verify_geometry, get_gate_debug_info, gpu_mercy_modulation,
    run_powrush_simulation_tick, should_trigger_mercy_evolution, GateEffects, GateEvent,
    MercyGeometryEvaluation, PowrushEntity, SimEntity,
};

fn platonic_zero() -> GeometryParams {
    GeometryParams {
        solid_type: SacredSolid::Platonic,
        dimensions: 3,
        symmetry_group: SymmetryGroup {
            order: 0,
            chiral: false,
        },
        evolution_step: 0,
        particle_density: 0.0,
        lattice_config: None,
    }
}

fn context(valence: f64) -> MercyContext {
    MercyContext {
        active_gates: vec![],
        valence,
        council_id: 4,
    }
}

fn score_at(value: f64) -> MercyGeometryEvaluation {
    MercyGeometryEvaluation {
        score: MercyGeometryScore {
            overall: value,
            love: value,
            mercy: value,
            truth: value,
            abundance: value,
            harmony: value,
            geometry_resonance: value,
        },
        should_evolve: false,
        should_apply_blessing: false,
        resonance: value,
        used_formal_path: false,
        all_gates_strong: false,
    }
}

#[test]
fn platonic_zero_matches_the_mial_source_literal_formulas() {
    let eval = evaluate_particle_geometry_mercy(&platonic_zero(), &context(0.999999));
    assert_eq!(eval.score.geometry_resonance, 0.96);
    assert_eq!(eval.score.harmony, 0.96);
    assert_eq!(eval.score.truth, 0.999);
    assert_eq!(
        eval.score.love,
        (0.96_f64 * 0.97 + 0.999999 * 0.03).min(1.0)
    );
    assert_eq!(eval.score.mercy, (0.96_f64 * 0.92 + 0.08).min(1.0));
    assert_eq!(eval.score.abundance, (0.96_f64 * 0.88 + 0.12).min(1.0));
    assert_eq!(
        eval.score.overall,
        (eval.score.love
            + eval.score.mercy
            + eval.score.truth
            + eval.score.abundance
            + eval.score.harmony)
            / 5.0
    );
    assert_eq!(eval.resonance, 0.96);
    assert!(eval.should_evolve);
    assert!(!eval.should_apply_blessing);
    assert!(eval.is_thriving());
    assert!(!eval.used_formal_path);
    assert!(!eval.all_gates_strong);
    assert_eq!(abundance_system_reaction(&eval), 1.08);
    assert_eq!(faction_harmony_reaction(&eval), 1.06);
    assert_eq!(gpu_mercy_modulation(&eval), 1.12);
    assert!(!evolution_system_reaction(&eval));
    assert!(!should_trigger_mercy_evolution(&eval));
    let sum = eval.score.love
        + eval.score.mercy
        + eval.score.truth
        + eval.score.abundance
        + eval.score.harmony
        + eval.score.geometry_resonance;
    assert_eq!(composite_gate_health(&eval), (sum / 6.0) as f32);
    assert!(eval
        .summary()
        .starts_with("MercyGeometry v14.15.0 | overall="));
    assert!(eval.summary().contains("evolve=true"));
    assert!(eval.summary().contains("blessing=false"));
    assert!(eval.summary().contains("formal=false"));
    assert!(eval.summary().contains("all_strong=false"));
}

#[test]
fn chiral_platonic_zero_truth_is_the_source_literal_0_995() {
    let mut geometry = platonic_zero();
    geometry.symmetry_group.chiral = true;
    let eval = evaluate_particle_geometry_mercy(&geometry, &context(0.999999));
    assert_eq!(eval.score.truth, 0.995);
}

#[test]
fn valence_below_the_source_literal_is_the_zero_sentinel() {
    let eval = evaluate_particle_geometry_mercy(&platonic_zero(), &context(0.999998));
    assert_eq!(eval.score.overall, 0.0);
    assert_eq!(eval.score.love, 0.0);
    assert_eq!(eval.score.mercy, 0.0);
    assert_eq!(eval.score.truth, 0.0);
    assert_eq!(eval.score.abundance, 0.0);
    assert_eq!(eval.score.harmony, 0.0);
    assert_eq!(eval.score.geometry_resonance, 0.0);
    assert_eq!(eval.resonance, 0.0);
    assert!(!eval.should_evolve);
    assert!(!eval.should_apply_blessing);
    assert!(!eval.is_thriving());
    assert_eq!(abundance_system_reaction(&eval), 0.85);
    assert_eq!(faction_harmony_reaction(&eval), 0.90);
    assert_eq!(gpu_mercy_modulation(&eval), 0.85);
}

#[test]
fn hyperbolic_zero_is_the_zero_sentinel() {
    let mut geometry = platonic_zero();
    geometry.solid_type = SacredSolid::Hyperbolic;
    let eval = evaluate_particle_geometry_mercy(&geometry, &context(0.999999));
    assert_eq!(eval.score.overall, 0.0);
    assert_eq!(eval.score.geometry_resonance, 0.0);
    assert!(!eval.should_evolve);
}

#[test]
fn no_bridge_keeps_the_mwpo_score() {
    let direct = evaluate_particle_geometry_mercy(&platonic_zero(), &context(0.999999));
    let formal = formally_verify_geometry(&platonic_zero(), &context(0.999999), None);
    assert_eq!(formal.score.overall, direct.score.overall);
    assert_eq!(formal.score.truth, 0.999);
    assert_eq!(formal.score.geometry_resonance, 0.96);
    assert!(!formal.used_formal_path);
    assert!(!formal.all_gates_strong);
    assert_eq!(formal.should_evolve, direct.should_evolve);
    assert_eq!(formal.should_apply_blessing, direct.should_apply_blessing);
}

#[test]
fn half_scores_stay_at_the_identity_multipliers() {
    let effects = compute_gate_effects(&score_at(0.5));
    assert_eq!(effects.resource_multiplier, 1.0);
    assert_eq!(effects.evolution_stability, 1.0);
    assert_eq!(effects.cooperation_bonus, 1.0);
    assert_eq!(effects.information_accuracy, 1.0);
    assert_eq!(effects.harmony_stability, 1.0);
    assert_eq!(effects.morale_bonus, 1.0);
    assert_eq!(effects.geometry_structural_bonus, 1.0);
    assert_eq!(GateEffects::identity().resource_multiplier, 1.0);
    assert_eq!(GateEffects::identity().geometry_structural_bonus, 1.0);
}

#[test]
fn zero_scores_take_the_low_score_penalty_literals() {
    let effects = compute_gate_effects(&score_at(0.0));
    assert_eq!(effects.resource_multiplier, 1.0);
    assert_eq!(effects.evolution_stability, 0.85);
    assert_eq!(effects.cooperation_bonus, 0.90);
    assert_eq!(effects.information_accuracy, 0.80);
    assert_eq!(effects.harmony_stability, 0.87);
    assert_eq!(effects.morale_bonus, 1.0 * 0.82 * 0.88);
    assert_eq!(effects.geometry_structural_bonus, 0.85);
}

#[test]
fn strong_flag_adds_the_source_literal_0_35_then_diminishes() {
    let mut eval = score_at(0.5);
    eval.all_gates_strong = true;
    let effects = compute_gate_effects(&eval);
    let raw = 1.0_f32 + 0.35;
    let diminished = 1.15 + ((raw - 1.15) * 0.4).min(0.6);
    assert_eq!(effects.evolution_stability, diminished);
    assert_eq!(effects.resource_multiplier, 1.0);
    assert_eq!(abundance_system_reaction(&eval), 1.25);
    assert_eq!(faction_harmony_reaction(&eval), 1.15);
    assert_eq!(gpu_mercy_modulation(&eval), 1.30);
    assert!(!evolution_system_reaction(&eval));
    eval.should_apply_blessing = true;
    assert!(evolution_system_reaction(&eval));
    assert_eq!(gpu_mercy_modulation(&score_at(0.5)), 0.85);
}

#[test]
fn confirmation_and_penalty_use_the_written_coefficients() {
    let eval = score_at(0.5);
    let base = compute_gate_effects(&eval);
    let same = apply_formal_confirmation_bonus(&eval, false);
    assert_eq!(same.evolution_stability, base.evolution_stability);
    assert_eq!(same.resource_multiplier, base.resource_multiplier);
    let up = apply_formal_confirmation_bonus(&eval, true);
    assert_eq!(up.evolution_stability, base.evolution_stability * 1.25);
    assert_eq!(up.resource_multiplier, base.resource_multiplier * 1.12);
    assert_eq!(
        up.geometry_structural_bonus,
        base.geometry_structural_bonus * 1.15
    );
    assert_eq!(up.morale_bonus, base.morale_bonus * 1.10);
    assert_eq!(up.cooperation_bonus, base.cooperation_bonus);
    assert_eq!(up.information_accuracy, base.information_accuracy);
    let pen = apply_formal_verification_failure_penalty(&eval);
    assert_eq!(
        pen.resource_multiplier,
        (base.resource_multiplier * 0.92).max(0.5)
    );
    assert_eq!(
        pen.evolution_stability,
        (base.evolution_stability * 0.88).max(0.4)
    );
    assert_eq!(
        pen.cooperation_bonus,
        (base.cooperation_bonus * 0.95).max(0.5)
    );
    assert_eq!(
        pen.information_accuracy,
        (base.information_accuracy * 0.90).max(0.5)
    );
    assert_eq!(
        pen.harmony_stability,
        (base.harmony_stability * 0.93).max(0.5)
    );
    assert_eq!(pen.morale_bonus, (base.morale_bonus * 0.94).max(0.5));
    assert_eq!(
        pen.geometry_structural_bonus,
        (base.geometry_structural_bonus * 0.91).max(0.5)
    );
}

#[test]
fn unit_score_emits_three_synergy_events() {
    let events = emit_and_collect_gate_events(&score_at(1.0), "e1");
    assert_eq!(events.len(), 3);
    match &events[0] {
        GateEvent::MajorSynergy {
            entity_id,
            synergy_type,
            bonus_magnitude,
            ..
        } => {
            assert_eq!(entity_id, "e1");
            assert_eq!(synergy_type, "Love + Harmony");
            assert_eq!(*bonus_magnitude, 1.18);
        }
        other => panic!("expected Love + Harmony, got {other:?}"),
    }
    match &events[1] {
        GateEvent::MajorSynergy {
            synergy_type,
            bonus_magnitude,
            ..
        } => {
            assert_eq!(synergy_type, "Truth + Abundance");
            assert_eq!(*bonus_magnitude, 1.15);
        }
        other => panic!("expected Truth + Abundance, got {other:?}"),
    }
    match &events[2] {
        GateEvent::MajorSynergy {
            synergy_type,
            bonus_magnitude,
            ..
        } => {
            assert_eq!(synergy_type, "Mercy + Overall");
            assert_eq!(*bonus_magnitude, 1.12);
        }
        other => panic!("expected Mercy + Overall, got {other:?}"),
    }
}

#[test]
fn low_scores_emit_two_penalty_events() {
    let events = emit_and_collect_gate_events(&score_at(0.3), "e");
    assert_eq!(events.len(), 2);
    match &events[0] {
        GateEvent::LowGatePenalty {
            entity_id,
            gate,
            value,
            ..
        } => {
            assert_eq!(entity_id, "e");
            assert_eq!(gate, "Love");
            assert_eq!(*value, 0.3_f64 as f32);
        }
        other => panic!("expected low love, got {other:?}"),
    }
    match &events[1] {
        GateEvent::LowGatePenalty { gate, value, .. } => {
            assert_eq!(gate, "Mercy");
            assert_eq!(*value, 0.3_f64 as f32);
        }
        other => panic!("expected low mercy, got {other:?}"),
    }
}

#[test]
fn strong_unit_score_adds_the_exceptional_event() {
    let mut eval = score_at(1.0);
    eval.all_gates_strong = true;
    let events = emit_and_collect_gate_events(&eval, "e");
    assert_eq!(events.len(), 4);
    match &events[3] {
        GateEvent::ExceptionalGateHealth {
            entity_id, health, ..
        } => {
            assert_eq!(entity_id, "e");
            assert_eq!(*health, 1.0);
        }
        other => panic!("expected exceptional health, got {other:?}"),
    }
}

#[test]
fn debug_info_names_the_dominant_field() {
    let harmony = get_gate_debug_info(&score_at(0.5), "e1", false, false);
    assert_eq!(harmony.entity_id, "e1");
    assert_eq!(harmony.dominant_gate, "Harmony");
    assert_eq!(harmony.overall_gate_health, 0.5);
    assert_eq!(harmony.synergy_level, 1.0);
    assert!(!harmony.diminishing_applied);
    assert!(!harmony.formal_path_used);
    assert!(!harmony.formal_success);

    let mut mercy = score_at(0.5);
    mercy.score.mercy = 0.9;
    mercy.score.love = 0.4;
    mercy.score.truth = 0.4;
    assert_eq!(
        get_gate_debug_info(&mercy, "e", false, false).dominant_gate,
        "Mercy"
    );

    let mut love = score_at(0.5);
    love.score.love = 0.9;
    love.score.mercy = 0.4;
    love.score.harmony = 0.4;
    love.score.geometry_resonance = 0.4;
    assert_eq!(
        get_gate_debug_info(&love, "e", false, false).dominant_gate,
        "Love"
    );

    let mut geometry = score_at(0.5);
    geometry.score.geometry_resonance = 0.9;
    assert_eq!(
        get_gate_debug_info(&geometry, "e", false, false).dominant_gate,
        "Geometry"
    );

    let mut high = score_at(1.0);
    high.score.overall = 1.2;
    let info = get_gate_debug_info(&high, "e", true, false);
    assert_eq!(info.synergy_level, 1.18);
    assert!(info.formal_path_used);
    assert!(!info.formal_success);
    assert!(info.diminishing_applied);
    assert_eq!(info.overall_gate_health, 1.2_f64 as f32);
}

#[test]
fn tick_without_a_bridge_applies_the_multiplier_and_leaves_stage() {
    let eval = score_at(0.5);
    let effects = compute_gate_effects(&eval);
    let mut entities = vec![
        PowrushEntity {
            id: "a".into(),
            resource_stock: 100.0,
            stability: 1.0,
            cooperation: 1.0,
            information_accuracy: 1.0,
            geometry_stability: 1.0,
            evolution_stage: 4,
            morale: 1.0,
        },
        PowrushEntity {
            id: "b".into(),
            resource_stock: 50.0,
            ..PowrushEntity::default()
        },
    ];
    run_powrush_simulation_tick(&mut entities, &[eval], None);
    assert_eq!(
        entities[0].resource_stock,
        (100.0 * effects.resource_multiplier).max(1.0)
    );
    assert_eq!(
        entities[0].stability,
        (1.0 * effects.evolution_stability).clamp(0.3, 1.8)
    );
    assert_eq!(
        entities[0].cooperation,
        (1.0 * effects.cooperation_bonus).clamp(0.2, 2.0)
    );
    assert_eq!(
        entities[0].information_accuracy,
        (1.0 * effects.information_accuracy).clamp(0.4, 1.6)
    );
    assert_eq!(
        entities[0].geometry_stability,
        (1.0 * effects.geometry_structural_bonus).clamp(0.3, 1.9)
    );
    assert_eq!(
        entities[0].morale,
        (1.0 * effects.morale_bonus).clamp(0.3, 2.0)
    );
    assert_eq!(entities[0].evolution_stage, 4);
    assert_eq!(entities[1].resource_stock, 50.0);
    assert_eq!(entities[1].id, "b");
}

#[test]
fn empty_tick_leaves_an_empty_slice() {
    let mut entities: [PowrushEntity; 0] = [];
    run_powrush_simulation_tick(&mut entities, &[], None);
    assert_eq!(entities.len(), 0);
}

#[test]
fn wrappers_use_the_half_score_identity_and_the_written_clamps() {
    let eval = score_at(0.5);
    assert_eq!(apply_resource_generation(&eval, 10.0), 10.0);
    assert_eq!(apply_resource_generation(&eval, 0.05), 0.1);
    assert_eq!(apply_cooperation_bonus(&eval, 1.5), 1.5);
    assert_eq!(apply_geometry_structural_bonus(&eval, 2.0), 2.0);
    assert_eq!(apply_information_accuracy(&eval, 2.0), 1.6);
    assert_eq!(apply_information_accuracy(&eval, 0.1), 0.4);
    assert_eq!(apply_evolution_stability(&eval, 2.0), 1.8);
    assert_eq!(apply_evolution_stability(&eval, 0.1), 0.3);
}

#[test]
fn default_entity_stock_is_the_source_literal_100() {
    let entity = PowrushEntity::default();
    let alias: SimEntity = entity.clone();
    assert_eq!(entity.id, "unknown");
    assert_eq!(entity.resource_stock, 100.0);
    assert_eq!(entity.stability, 1.0);
    assert_eq!(entity.evolution_stage, 0);
    assert_eq!(alias.resource_stock, 100.0);
    assert_eq!(alias.id, "unknown");
}

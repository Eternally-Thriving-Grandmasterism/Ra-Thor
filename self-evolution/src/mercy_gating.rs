//! Connected SelfEvolutionBlessing to existing blessing system

// ... existing code ...

/// Enhanced evaluation for SelfEvolutionBlessing that can consider blessing system state
pub fn evaluate_self_evolution_blessing(
    base_score: f64,
    current_blessing_level: f64,
    recent_blessing_success_rate: f64,
) -> MercyVerdict {
    let adjusted_score = base_score 
        + (current_blessing_level * 0.08)
        + (recent_blessing_success_rate * 0.06);

    let final_score = adjusted_score.min(0.999);

    if final_score >= 0.88 {
        MercyVerdict::Mitigated {
            overall_score: final_score,
            notes: vec![
                format!("Self-Evolution Blessing: Adjusted score {:.3} (blessing synergy applied)", final_score)
            ],
        }
    } else {
        MercyVerdict::RequiresCouncilReview
    }
}

// We can later wire this into SovereignHealthMonitor.request_epigenetic_blessing()

// ... rest of file ...
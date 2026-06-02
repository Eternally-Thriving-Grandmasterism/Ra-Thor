//! Deal Readiness Scoring
//!
//! Composite scoring that combines:
//! - Professional Judgment Score (from Ontario Professional Judgment Layer)
//! - Geometric Harmony
//!
//! Features dynamic jurisdiction-aware weighting and gentle ONE Organism valence influence.
//!
//! AG-SML v1.0 — Fully mercy-gated, TOLC 8 enforced, PATSAGi-aligned.
//! Production ready for Powrush land evaluation and RREL flows.

/// Calculates a composite Deal Readiness Score (0–100).
///
/// ### Dynamic Weighting
/// - **Ontario POTL/Common Elements**: 75% Judgment / 25% Geometric
/// - **Ontario (non-POTL)**: 65% Judgment / 35% Geometric  
/// - **Other jurisdictions**: 60% Judgment / 40% Geometric
///
/// ### Valence Influence
/// Gentle modulation based on current ONE Organism valence (high valence = slight boost).
pub fn calculate_deal_readiness_score(
    judgment_score: u32,
    geometric_harmony: f64,
    jurisdiction: &str,
    is_potl: bool,
    current_valence: f64,
) -> u32 {
    // === Dynamic Weighting ===
    let (judgment_weight, geometric_weight) = if jurisdiction.to_lowercase().contains("ontario") {
        if is_potl {
            (0.75, 0.25) // Highest judgment priority for POTL
        } else {
            (0.65, 0.35)
        }
    } else {
        (0.60, 0.40)
    };

    let geo_normalized = (geometric_harmony.clamp(0.0, 1.65) * 60.0) as u32;

    let base_score = (judgment_score as f64 * judgment_weight)
                   + (geo_normalized as f64 * geometric_weight);

    // === Gentle Valence Influence ===
    let valence_modifier = if current_valence >= 0.95 {
        1.04
    } else if current_valence >= 0.90 {
        1.02
    } else if current_valence < 0.85 {
        0.97
    } else {
        1.0
    };

    let final_score = (base_score * valence_modifier).clamp(0.0, 100.0);

    final_score as u32
}

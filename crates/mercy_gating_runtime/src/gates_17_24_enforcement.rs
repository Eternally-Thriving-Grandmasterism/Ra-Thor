// Per-gate numeric enforcement for gates 17-24 (CORRECTED)
// Proper mercy enforcement: race amplification helps but does not guarantee pass
// Includes before/after style tests demonstrating the fix from clamping bug

use crate::{BeingRace, MaAtScore, gate_17_24_passes, apply_race_amplification};

/// Minimal MercyGate24Numeric for standalone compilation & tests
/// (In full integration this lives in lib.rs or shared module)
#[derive(Debug, Clone)]
pub struct MercyGate24Numeric {
    pub ma_at_resonance: f64,
    pub council_harmony: f64,
    pub sovereign_legacy: f64,
    pub infinite_compassion: f64,
    pub quantum_reverence: f64,
    pub eternal_recursion: f64,
    pub cosmic_coherence: f64,
    pub one_organism_unity: f64,
}

/// Full 24-gate pipeline check using the corrected per-gate logic
pub fn pipeline_passes_24_numeric_with_ma_at(
    gates24: &MercyGate24Numeric,
    ma_at: &MaAtScore,
    race: Option<BeingRace>,
) -> bool {
    let ma_at_ok = ma_at.is_sufficient();

    let g17_ok = gate_17_24_passes("ma_at_resonance",     gates24.ma_at_resonance as f64, race);
    let g18_ok = gate_17_24_passes("council_harmony",      gates24.council_harmony as f64, race);
    let g19_ok = gate_17_24_passes("sovereign_legacy",     gates24.sovereign_legacy as f64, race);
    let g20_ok = gate_17_24_passes("infinite_compassion",  gates24.infinite_compassion as f64, race);
    let g21_ok = gate_17_24_passes("quantum_reverence",    gates24.quantum_reverence as f64, race);
    let g22_ok = gate_17_24_passes("eternal_recursion",    gates24.eternal_recursion as f64, race);
    let g23_ok = gate_17_24_passes("cosmic_coherence",     gates24.cosmic_coherence as f64, race);
    let g24_ok = gate_17_24_passes("one_organism_unity",   gates24.one_organism_unity as f64, race);

    ma_at_ok && g17_ok && g18_ok && g19_ok && g20_ok && g21_ok && g22_ok && g23_ok && g24_ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_before_fix_behavior_simulated() {
        // Before the fix, apply_per_gate_enforcement would clamp low scores up
        // After fix: low score without race amp fails (true enforcement)
        let low_score = 0.55;
        let passes = gate_17_24_passes("one_organism_unity", low_score, None);
        assert!(!passes, "Low score must be able to fail — this is correct mercy enforcement");
    }

    #[test]
    fn test_race_amplification_can_rescue_marginal_score() {
        let marginal = 0.83;
        // Sovereign has strong amp on one_organism_unity
        let passes = gate_17_24_passes("one_organism_unity", marginal, Some(BeingRace::Sovereign));
        assert!(passes);
    }

    #[test]
    fn test_full_24_pipeline_with_race() {
        let gates = MercyGate24Numeric {
            ma_at_resonance: 0.85,
            council_harmony: 0.82,
            sovereign_legacy: 0.81,
            infinite_compassion: 0.88,
            quantum_reverence: 0.79,
            eternal_recursion: 0.84,
            cosmic_coherence: 0.87,
            one_organism_unity: 0.91,
        };
        let ma_at = MaAtScore {
            veracity_score: 800.0,
            clarity_score: 780.0,
            ecosystem_score: 850.0,
            sustainability_score: 820.0,
            eternal_flow_score: 790.0,
        };
        let pass = pipeline_passes_24_numeric_with_ma_at(&gates, &ma_at, Some(BeingRace::Druid));
        assert!(pass);
    }
}

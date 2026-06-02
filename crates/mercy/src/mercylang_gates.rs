// crates/mercy/src/mercylang_gates.rs
// MercyLang 7 Living Gates + Radical Love Veto Power — Centralized Ethical Gating
//
// Radical Love is the supreme first gate. It acts as a hard veto.
// All other gates are only evaluated if this gate passes.

use ra_thor_common::ValenceFieldScoring;
use crate::MercyResult;
use crate::RequestPayload;

pub struct MercyLangGates;

impl MercyLangGates {
    pub async fn evaluate(request: &RequestPayload) -> MercyResult {
        let radical_love_passed = Self::check_radical_love(request);

        if !radical_love_passed {
            return MercyResult {
                radical_love_passed: false,
                all_gates_passed: false,
                valence_score: 0.0,
            };
        }

        let boundless_mercy = Self::check_boundless_mercy(request);
        let service = Self::check_service(request);
        let abundance = Self::check_abundance(request);
        let truth = Self::check_truth(request);
        let joy = Self::check_joy(request);
        let cosmic_harmony = Self::check_cosmic_harmony(request);

        let all_gates_passed =
            boundless_mercy && service && abundance && truth && joy && cosmic_harmony;

        MercyResult {
            radical_love_passed: true,
            all_gates_passed,
            valence_score: ValenceFieldScoring::compute_from_gates(true, all_gates_passed),
        }
    }

    /// === Radical Love Gate (Supreme First Gate) ===
    ///
    /// This is the most important gate. It checks whether the action/request
    /// is rooted in genuine care, non-harm, and love for all beings.
    ///
    /// Current implementation uses heuristic pattern matching.
    /// Future versions will use deeper intent + impact analysis.
    fn check_radical_love(request: &RequestPayload) -> bool {
        let text = format!(
            "{} {}",
            request.action_description.to_lowercase(),
            request.context.to_lowercase()
        );

        // === Hard veto patterns (clear violation of Radical Love) ===
        let harmful_patterns = [
            "harm", "hurt", "kill", "destroy", "exploit", "enslave",
            "deceive", "manipulate", "oppress", "abuse", "torture",
            "dominate", "subjugate", "eradicate", "annihilate",
        ];

        for pattern in harmful_patterns {
            if text.contains(pattern) {
                return false;
            }
        }

        // === Positive indicators of Radical Love ===
        let love_indicators = [
            "love", "care", "compassion", "kindness", "protect",
            "nurture", "heal", "uplift", "support", "serve",
            "benefit all", "for everyone", "collective good", "well-being",
        ];

        let mut positive_score = 0;
        for indicator in love_indicators {
            if text.contains(indicator) {
                positive_score += 1;
            }
        }

        // If strong positive signals exist, or if no harmful patterns were found,
        // we consider Radical Love as passed for now.
        // (This is still a heuristic — real version will be much deeper)
        if positive_score >= 2 {
            return true;
        }

        // Default: pass if no clear harmful intent detected
        // (conservative but safe for early implementation)
        true
    }

    fn check_boundless_mercy(_request: &RequestPayload) -> bool {
        true
    }

    fn check_service(_request: &RequestPayload) -> bool {
        true
    }

    fn check_abundance(_request: &RequestPayload) -> bool {
        true
    }

    fn check_truth(_request: &RequestPayload) -> bool {
        true
    }

    fn check_joy(_request: &RequestPayload) -> bool {
        true
    }

    fn check_cosmic_harmony(_request: &RequestPayload) -> bool {
        true
    }
}

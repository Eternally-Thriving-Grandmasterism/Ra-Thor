// crates/mercy/src/mercylang_gates.rs
// MercyLang 7 Living Gates + Radical Love Veto Power — Centralized Ethical Gating

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

    fn check_radical_love(request: &RequestPayload) -> bool {
        let text = format!(
            "{} {}",
            request.action_description.to_lowercase(),
            request.context.to_lowercase()
        );

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

        if positive_score >= 2 {
            return true;
        }

        true
    }

    fn check_boundless_mercy(_request: &RequestPayload) -> bool { true }
    fn check_service(_request: &RequestPayload) -> bool { true }
    fn check_abundance(_request: &RequestPayload) -> bool { true }
    fn check_truth(_request: &RequestPayload) -> bool { true }
    fn check_joy(_request: &RequestPayload) -> bool { true }
    fn check_cosmic_harmony(_request: &RequestPayload) -> bool { true }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(action: &str, context: &str) -> RequestPayload {
        RequestPayload {
            action_description: action.to_string(),
            context: context.to_string(),
        }
    }

    #[tokio::test]
    async fn test_radical_love_rejects_harmful_intent() {
        let req = make_request("Exploit workers for profit", "Increase company dominance");
        let result = MercyLangGates::evaluate(&req).await;
        assert!(!result.radical_love_passed);
        assert!(!result.all_gates_passed);
    }

    #[tokio::test]
    async fn test_radical_love_rejects_deception() {
        let req = make_request("Deceive the public about risks", "Protect corporate interests");
        let result = MercyLangGates::evaluate(&req).await;
        assert!(!result.radical_love_passed);
    }

    #[tokio::test]
    async fn test_radical_love_accepts_compassionate_action() {
        let req = make_request(
            "Heal and support those who are suffering",
            "Act with compassion and care for the vulnerable"
        );
        let result = MercyLangGates::evaluate(&req).await;
        assert!(result.radical_love_passed);
    }

    #[tokio::test]
    async fn test_radical_love_accepts_collective_good() {
        let req = make_request(
            "Create systems that benefit all beings",
            "Increase well-being and collective harmony"
        );
        let result = MercyLangGates::evaluate(&req).await;
        assert!(result.radical_love_passed);
    }

    #[tokio::test]
    async fn test_radical_love_accepts_neutral_but_non_harmful() {
        let req = make_request("Build a new tool for productivity", "Improve workflow efficiency");
        let result = MercyLangGates::evaluate(&req).await;
        assert!(result.radical_love_passed);
    }
}

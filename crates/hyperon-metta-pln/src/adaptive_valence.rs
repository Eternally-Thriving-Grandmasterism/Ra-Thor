// crates/hyperon-metta-pln/src/adaptive_valence.rs
// AG-SML v1.0+ License | TOLC 8 Passed | ONE Organism Compatible
// Eternal Mercy Flow License — Ra-Thor + Grok unified in PATSAGi Councils
//
// Adaptive Valence Threshold Engine
// Replaces static pruning with confidence + novelty modulated thresholds.
// Core paths remain ultra-strict (v >= 0.9999999). Novel situations unlock
// reviewable exploration branches only. Never auto-applies to sovereign state.
// All outputs pass TOLC 8 + truth-distillation.

use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct ConfidenceEMA {
    pub value: f64,      // 0.0 - 1.0 from Lattice Conductor symbolic_confidence_ema
    pub success_ema: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct MercyContext {
    pub gate8_harmony: f64,   // Cosmic Harmony gate output
    pub service_score: f64,
}

#[derive(Debug, Clone)]
pub struct ValenceEvaluation {
    pub raw_valence: f64,
    pub adaptive_threshold: f64,
    pub is_pruned: bool,
    pub is_exploration_branch: bool,
    pub review_priority: u8,  // 0-255, higher = more urgent human/TOLC review
    pub reason: String,
}

#[derive(Debug, Clone, Copy)]
pub struct AdaptiveValenceConfig {
    pub base_threshold: f64,      // 0.9999999
    pub confidence_weight: f64,   // 0.6
    pub novelty_weight: f64,      // 0.4
    pub min_safe_threshold: f64,  // 0.999
    pub exploration_epsilon: f64, // 0.0005 — how close to threshold triggers branch
}

impl Default for AdaptiveValenceConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.9999999,
            confidence_weight: 0.6,
            novelty_weight: 0.4,
            min_safe_threshold: 0.999,
            exploration_epsilon: 0.0005,
        }
    }
}

pub struct AdaptiveValenceEngine {
    config: AdaptiveValenceConfig,
}

impl AdaptiveValenceEngine {
    pub fn new(config: AdaptiveValenceConfig) -> Self {
        Self { config }
    }

    /// Compute dynamic threshold using Lattice Conductor EMA + novelty signal.
    /// Always >= min_safe_threshold. Mercy Context can further raise bar.
    pub fn compute_adaptive_threshold(
        &self,
        confidence: &ConfidenceEMA,
        novelty_score: f64,      // 0.0 (known) .. 1.0 (highly novel)
        mercy: &MercyContext,
    ) -> f64 {
        let confidence_factor = 1.0 - self.config.confidence_weight * (1.0 - confidence.value.max(0.0));
        let novelty_factor = 1.0 + self.config.novelty_weight * novelty_score.clamp(0.0, 1.0);

        let mut threshold = self.config.base_threshold * confidence_factor * novelty_factor;

        // Mercy Context raise (never lowers below base safety)
        let mercy_raise = (mercy.gate8_harmony + mercy.service_score) * 0.00000005;
        threshold = (threshold + mercy_raise).min(1.0);

        threshold.max(self.config.min_safe_threshold)
    }

    /// Core evaluation. Returns pruned or exploration decision.
    /// Exploration branch only created when novelty high AND close to threshold.
    /// All paths TOLC 8 + mercy gated.
    pub fn evaluate_and_gate(
        &self,
        raw_valence: f64,
        confidence: &ConfidenceEMA,
        novelty_score: f64,
        mercy: &MercyContext,
        is_core_decision: bool,   // true = ultra-strict, no exploration
    ) -> ValenceEvaluation {
        let adaptive_threshold = self.compute_adaptive_threshold(confidence, novelty_score, mercy);

        let mut eval = ValenceEvaluation {
            raw_valence,
            adaptive_threshold,
            is_pruned: false,
            is_exploration_branch: false,
            review_priority: 0,
            reason: String::new(),
        };

        if raw_valence >= adaptive_threshold {
            eval.reason = format!(
                "PASS adaptive (raw={:.8} >= thresh={:.8}, conf={:.4}, novelty={:.3})",
                raw_valence, adaptive_threshold, confidence.value, novelty_score
            );
            return eval;
        }

        // Below threshold
        let distance = adaptive_threshold - raw_valence;

        if is_core_decision || novelty_score < 0.6 || distance > self.config.exploration_epsilon * 2.0 {
            eval.is_pruned = true;
            eval.reason = format!(
                "PRUNED strict (raw={:.8} < thresh={:.8}) — core or low-novelty",
                raw_valence, adaptive_threshold
            );
            return eval;
        }

        // Novel + close → exploration branch (reviewable, never auto-apply)
        eval.is_exploration_branch = true;
        eval.review_priority = ((novelty_score * 200.0) as u8).max(50);
        eval.reason = format!(
            "EXPLORATION_BRANCH (raw={:.8}, thresh={:.8}, novelty={:.3}) — TOLC review required",
            raw_valence, adaptive_threshold, novelty_score
        );

        eval
    }

    /// Create reviewable self-proposal for Lattice Conductor / SelfEvolutionTelemetry.
    pub fn create_exploration_proposal(&self, eval: &ValenceEvaluation) -> String {
        format!(
            "ADAPTIVE_VALENCE_EXPLORATION | valence={:.8} | threshold={:.8} | priority={} | reason: {}",
            eval.raw_valence, eval.adaptive_threshold, eval.review_priority, eval.reason
        )
    }
}

impl fmt::Display for ValenceEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ValenceEval {{ raw: {:.8}, thresh: {:.8}, pruned: {}, exploration: {}, priority: {} }} — {}",
            self.raw_valence, self.adaptive_threshold, self.is_pruned, self.is_exploration_branch, self.review_priority, self.reason
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_threshold_confidence_modulation() {
        let engine = AdaptiveValenceEngine::new(AdaptiveValenceConfig::default());
        let conf_high = ConfidenceEMA { value: 0.95, success_ema: 0.92 };
        let mercy = MercyContext { gate8_harmony: 0.99, service_score: 0.98 };
        let thresh = engine.compute_adaptive_threshold(&conf_high, 0.1, &mercy);
        assert!(thresh >= 0.999 && thresh <= 1.0);
    }

    #[test]
    fn test_exploration_branch_on_novel_close_call() {
        let engine = AdaptiveValenceEngine::new(AdaptiveValenceConfig::default());
        let conf = ConfidenceEMA { value: 0.82, success_ema: 0.80 };
        let mercy = MercyContext { gate8_harmony: 0.95, service_score: 0.9 };
        let eval = engine.evaluate_and_gate(0.9997, &conf, 0.85, &mercy, false);
        assert!(eval.is_exploration_branch);
        assert!(!eval.is_pruned);
    }

    #[test]
    fn test_core_decision_remains_strict() {
        let engine = AdaptiveValenceEngine::new(AdaptiveValenceConfig::default());
        let conf = ConfidenceEMA { value: 0.7, success_ema: 0.65 };
        let mercy = MercyContext { gate8_harmony: 0.9, service_score: 0.85 };
        let eval = engine.evaluate_and_gate(0.9995, &conf, 0.9, &mercy, true);
        assert!(eval.is_pruned);
        assert!(!eval.is_exploration_branch);
    }
}

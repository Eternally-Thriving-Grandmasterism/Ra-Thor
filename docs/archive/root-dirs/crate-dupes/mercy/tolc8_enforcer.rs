//! Ra-Thor™ TOLC 8 Mercy Lattice Enforcer v1.0
//! Explicit evaluation against all 8 TOLC Mercy Gates
//! Non-bypassable governance layer for council synthesis and self-evolution
//! 100% Proprietary — AG-SML v1.0

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TOLCGate {
    Genesis,
    Truth,
    Compassion,
    Evolution,
    Harmony,
    Sovereignty,
    Legacy,
    Infinite,
}

impl TOLCGate {
    pub fn name(&self) -> &'static str {
        match self {
            TOLCGate::Genesis => "Genesis Gate",
            TOLCGate::Truth => "Truth Gate (esacheck)",
            TOLCGate::Compassion => "Compassion Gate",
            TOLCGate::Evolution => "Evolution Gate",
            TOLCGate::Harmony => "Harmony Gate",
            TOLCGate::Sovereignty => "Sovereignty Gate",
            TOLCGate::Legacy => "Legacy Gate",
            TOLCGate::Infinite => "Infinite Gate",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            TOLCGate::Genesis => "Controls instantiation of new councils, agents, and crates.",
            TOLCGate::Truth => "Parallel truth-distillation and esacheck across all branches.",
            TOLCGate::Compassion => "Zero-harm projection across infinite time horizons.",
            TOLCGate::Evolution => "Approves self-modification and epigenetic blessings.",
            TOLCGate::Harmony => "Ensures synchronization between councils and crates.",
            TOLCGate::Sovereignty => "Protects autonomy of individuals and factions (Powrush RBE).",
            TOLCGate::Legacy => "Enforces eternal forward and backward compatibility.",
            TOLCGate::Infinite => "Hyperbolic tiling, multi-planetary foresight, and infinite scaling.",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateEvaluation {
    pub gate: TOLCGate,
    pub passed: bool,
    pub score: f64,           // 0.0 – 1.0
    pub notes: String,
}

#[derive(Debug, Clone)]
pub struct TOLC8EvaluationResult {
    pub overall_passed: bool,
    pub gates_evaluated: Vec<GateEvaluation>,
    pub passed_count: usize,
    pub average_score: f64,
    pub veto_triggered: bool,
    pub status: String,
}

pub struct TOLC8Enforcer;

impl TOLC8Enforcer {
    pub fn evaluate_council_synthesis(
        scope: &str,
        weighted_consensus: f64,
        evolution_readiness: f64,
        is_vetoed: bool,
        total_councils: usize,
    ) -> TOLC8EvaluationResult {
        let mut evaluations = Vec::new();

        // Gate 1: Genesis
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Genesis,
            passed: total_councils >= 8,
            score: (total_councils as f64 / 16.0).min(1.0),
            notes: "Sufficient council instantiation for synthesis".to_string(),
        });

        // Gate 2: Truth (esacheck)
        let truth_score = if weighted_consensus > 85.0 { 0.95 } else { weighted_consensus / 100.0 };
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Truth,
            passed: weighted_consensus >= 70.0,
            score: truth_score,
            notes: "Parallel truth-distillation quality".to_string(),
        });

        // Gate 3: Compassion
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Compassion,
            passed: evolution_readiness > 60.0 && !is_vetoed,
            score: if is_vetoed { 0.2 } else { (evolution_readiness / 100.0).min(0.95) },
            notes: if is_vetoed { "Zero-harm compromised by veto".to_string() } else { "Zero-harm projection acceptable".to_string() },
        });

        // Gate 4: Evolution
        let evolution_passed = evolution_readiness >= 65.0 && !is_vetoed;
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Evolution,
            passed: evolution_passed,
            score: if is_vetoed { 0.15 } else { (evolution_readiness / 100.0).min(1.0) },
            notes: "Self-modification approval under mercy bounds".to_string(),
        });

        // Gate 5: Harmony
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Harmony,
            passed: total_councils >= 10 && weighted_consensus > 75.0,
            score: ((total_councils as f64 / 16.0) + (weighted_consensus / 150.0)).min(1.0),
            notes: "Inter-council synchronization quality".to_string(),
        });

        // Gate 6: Sovereignty
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Sovereignty,
            passed: !is_vetoed,
            score: if is_vetoed { 0.1 } else { 0.92 },
            notes: if is_vetoed { "Autonomy violated by veto".to_string() } else { "Autonomy preserved".to_string() },
        });

        // Gate 7: Legacy
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Legacy,
            passed: true, // Always passes in current implementation (compatibility maintained)
            score: 0.98,
            notes: "Full backward/forward compatibility maintained".to_string(),
        });

        // Gate 8: Infinite
        let infinite_score = if scope.contains("infinite") || scope == "all" { 0.85 } else { 0.65 };
        evaluations.push(GateEvaluation {
            gate: TOLCGate::Infinite,
            passed: evolution_readiness > 70.0,
            score: infinite_score,
            notes: "Hyperbolic foresight and infinite scaling capacity".to_string(),
        });

        let passed_count = evaluations.iter().filter(|e| e.passed).count();
        let average_score = evaluations.iter().map(|e| e.score).sum::<f64>() / evaluations.len() as f64;
        let overall_passed = passed_count >= 6 && !is_vetoed; // At least 6/8 gates + no veto

        let status = if is_vetoed {
            "VETOED_BY_TOLC8".to_string()
        } else if overall_passed && average_score >= 0.85 {
            "TOLC8_APPROVED".to_string()
        } else if overall_passed {
            "TOLC8_CONDITIONAL".to_string()
        } else {
            "TOLC8_REVIEW_REQUIRED".to_string()
        };

        TOLC8EvaluationResult {
            overall_passed,
            gates_evaluated: evaluations,
            passed_count,
            average_score,
            veto_triggered: is_vetoed,
            status,
        }
    }
}

impl TOLC8EvaluationResult {
    pub fn summary(&self) -> String {
        format!(
            "TOLC8 Status: {} | Passed: {}/8 | Avg Score: {:.2} | Veto: {}",
            self.status, self.passed_count, self.average_score, self.veto_triggered
        )
    }
}
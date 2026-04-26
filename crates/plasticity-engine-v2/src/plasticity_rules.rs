//! Plasticity Rules Engine
//! Defines and evaluates the 4 core plasticity rules for the 5-Gene Joy Tetrad

pub struct PlasticityRulesEngine;

impl PlasticityRulesEngine {
    pub fn new() -> Self {
        Self
    }

    pub async fn evaluate(
        &self,
        impact: &crate::legal_lattice::cehi::CEHIImpact,
    ) -> Result<RuleResult, crate::PlasticityError> {
        if impact.improvement >= 0.25 {
            Ok(RuleResult {
                rule_name: "JoyTetradLock".to_string(),
                should_apply: true,
                strength: impact.improvement,
            })
        } else if impact.improvement >= 0.15 {
            Ok(RuleResult {
                rule_name: "MetaplasticReinforcement".to_string(),
                should_apply: true,
                strength: impact.improvement * 0.8,
            })
        } else {
            Ok(RuleResult {
                rule_name: "HomeostaticMaintenance".to_string(),
                should_apply: true,
                strength: impact.improvement * 0.6,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuleResult {
    pub rule_name: String,
    pub should_apply: bool,
    pub strength: f64,
}

// crates/ra-thor-meta-intelligence/src/self_improvement_engine.rs

use crate::crate_analyzer::{CrateAnalyzer, CrateHealthReport};
use mercy_merlin_engine::{MercyMerlinEngine, MercyEvaluationResult};
use async_trait::async_trait;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum ImprovementType {
    MercyIntegration,
    TOLCAlignment,
    SelfImprovementCapability,
    Documentation,
    CryptographyHardening,
    CrossCrateCoordination,
    Testing,
    ArchitectureImprovement,
    Maintenance,
}

#[derive(Debug, Clone)]
pub struct ImprovementProposal {
    pub id: String,
    pub target_crate: String,
    pub improvement_type: ImprovementType,
    pub title: String,
    pub description: String,
    pub expected_impact: u8,
    pub mercy_alignment: u8,
    pub implementation_difficulty: u8,
    pub priority_score: f64,
    pub rationale: String,
}

#[derive(Debug)]
pub struct MercyGateResult {
    pub passed: bool,
    pub current_valence: f64,
    pub council_consensus: bool,
    pub rejection_reason: Option<String>,
}

pub struct SelfImprovementEngine {
    mercy_engine: MercyMerlinEngine,
    crate_analyzer: CrateAnalyzer,
    pub recent_proposals: Vec<ImprovementProposal>,
    pub improvement_history: Vec<ImprovementProposal>,
    pub total_suggestions_made: u64,
}

impl SelfImprovementEngine {
    pub fn new(mercy_engine: MercyMerlinEngine, crate_analyzer: CrateAnalyzer) -> Self {
        Self {
            mercy_engine,
            crate_analyzer,
            recent_proposals: Vec::new(),
            improvement_history: Vec::new(),
            total_suggestions_made: 0,
        }
    }

    pub async fn check_mercy_gate(&self) -> Result<MercyGateResult, Box<dyn std::error::Error + Send + Sync>> {
        let valence = self.mercy_engine.get_current_valence().await?;
        let consensus_result = self.mercy_engine.evaluate_council_consensus().await?;

        let passed = valence >= 0.92 && consensus_result.approved;

        Ok(MercyGateResult {
            passed,
            current_valence: valence,
            council_consensus: consensus_result.approved,
            rejection_reason: if !passed {
                Some(format!("Valence: {:.4}, Council Approved: {}", valence, consensus_result.approved))
            } else {
                None
            },
        })
    }

    pub async fn generate_improvement_proposals(
        &mut self,
    ) -> Result<Vec<ImprovementProposal>, Box<dyn std::error::Error + Send + Sync>> {

        let mercy_result = self.check_mercy_gate().await?;

        if !mercy_result.passed {
            tracing::warn!("Mercy gate rejected proposal generation. Reason: {:?}", mercy_result.rejection_reason);
            return Ok(vec![]);
        }

        let health_reports = self.crate_analyzer.analyze_critical_crates().await?;
        let mut proposals = Vec::new();

        for report in health_reports {
            if report.importance >= 8 && report.mercy_integration_score < 6 {
                proposals.push(self.create_proposal(&report, ImprovementType::MercyIntegration, "Add proper mercy_merlin_engine valence checks and council consensus", 9, 10, 5));
            }
            if report.importance >= 8 && report.technical_debt_score >= 7 {
                proposals.push(self.create_proposal(&report, ImprovementType::ArchitectureImprovement, "Refactor to reduce technical debt and improve long-term maintainability", 8, 8, 6));
            }
            if report.importance >= 8 && report.test_coverage_score < 5 {
                proposals.push(self.create_proposal(&report, ImprovementType::Testing, "Significantly improve test coverage and add integration tests", 8, 9, 4));
            }
            if report.importance >= 7 && report.documentation_score < 5 {
                proposals.push(self.create_proposal(&report, ImprovementType::Documentation, "Improve documentation, examples, and API docs", 6, 8, 3));
            }
            if report.importance >= 8 && report.last_activity_days > 60 {
                proposals.push(self.create_proposal(&report, ImprovementType::Maintenance, "Review and modernize stale high-importance crate", 7, 7, 5));
            }
            if report.importance >= 7 && report.mercy_integration_score < 5 && report.technical_debt_score >= 6 && report.test_coverage_score < 5 {
                proposals.push(self.create_proposal(&report, ImprovementType::ArchitectureImprovement, "Comprehensive modernization: mercy integration + debt reduction + testing", 9, 9, 7));
            }
        }

        for proposal in &mut proposals {
            proposal.priority_score = SelfImprovementEngine::calculate_priority_score(proposal);
        }

        let mut high_quality: Vec<ImprovementProposal> = proposals.into_iter()
            .filter(|p| p.mercy_alignment >= 8 && p.priority_score >= 7.0)
            .collect();

        high_quality.sort_by(|a, b| b.priority_score.partial_cmp(&a.priority_score).unwrap());

        self.recent_proposals = high_quality.clone();
        self.total_suggestions_made += high_quality.len() as u64;

        Ok(high_quality)
    }

    fn create_proposal(
        &self,
        report: &CrateHealthReport,
        improvement_type: ImprovementType,
        action: &str,
        impact: u8,
        mercy: u8,
        difficulty: u8,
    ) -> ImprovementProposal {
        ImprovementProposal {
            id: format!("imp-{}", self.total_suggestions_made),
            target_crate: report.crate_name.clone(),
            improvement_type,
            title: format!("{} in {}", action, report.crate_name),
            description: format!("{} (Mercy: {}, Debt: {}, Tests: {}, Docs: {})", action, report.mercy_integration_score, report.technical_debt_score, report.test_coverage_score, report.documentation_score),
            expected_impact: impact,
            mercy_alignment: mercy,
            implementation_difficulty: difficulty,
            priority_score: 0.0,
            rationale: format!("High-importance crate ({}) with identified weaknesses.", report.importance),
        }
    }

    fn calculate_priority_score(proposal: &ImprovementProposal) -> f64 {
        (proposal.expected_impact as f64 * 0.40) + (proposal.mercy_alignment as f64 * 0.35) + ((10 - proposal.implementation_difficulty) as f64 * 0.25)
    }
}
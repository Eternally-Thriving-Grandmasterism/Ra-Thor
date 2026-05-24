/// Advanced Orchestrator
///
/// With WCAG AA accessibility scoring integrated into results.

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::ComponentTree;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::renderer::render_tree;
use crate::orchestration::semantic_planning::{SemanticPlanningStrategy, OpenAIEmbeddingProvider};
use crate::validation::{HtmlValidator, calculate_wcag_aa_score, WcagAaScore};
use serde_json::from_str;
use std::time::Instant;
use tracing::info_span;

#[derive(Debug)]
pub struct AdvancedOrchestrationResult {
    pub final_html: Option<String>,
    pub component_tree: Option<ComponentTree>,
    pub validation_issues: Vec<String>,
    pub attempts_used: usize,
    pub success: bool,
    pub wcag_aa_score: Option<WcagAaScore>,   // NEW: Accessibility score
}

// ... (PlanningResult etc. unchanged)

impl AdvancedOrchestrator {
    pub fn orchestrate(&self, prompt: &str) -> AdvancedOrchestrationResult {
        let start = Instant::now();
        let _root_span = info_span!("orchestration", prompt = %prompt).entered();

        // ... orchestration logic ...

        // After generation/validation
        let final_html = None; // placeholder from actual generation
        let wcag_score = final_html.as_ref().map(|html| calculate_wcag_aa_score(html));

        let result = AdvancedOrchestrationResult {
            final_html,
            component_tree: None,
            validation_issues: vec![],
            attempts_used: 1,
            success: false,
            wcag_aa_score: wcag_score,
        };

        let duration = start.elapsed().as_secs_f64();
        observability::record_orchestration_metrics(duration, result.success, result.attempts_used);

        result
    }
}

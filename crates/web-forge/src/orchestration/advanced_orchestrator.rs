/// Advanced Orchestrator
///
/// With deep refinement instrumentation.

use crate::observability;
use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::ComponentTree;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::renderer::render_tree;
use crate::orchestration::semantic_planning::{SemanticPlanningStrategy, OpenAIEmbeddingProvider};
use crate::validation::HtmlValidator;
use serde_json::from_str;
use std::time::Instant;
use tracing::{info_span, warn};

// ... (rest of the code)

impl AdvancedOrchestrator {
    pub fn orchestrate(&self, prompt: &str) -> AdvancedOrchestrationResult {
        let start = Instant::now();
        let _root_span = info_span!("orchestration", prompt = %prompt).entered();

        // Planning + Generation...

        // === Refinement Loop with Deep Instrumentation ===
        let refinement_span = info_span!("refinement_loop");
        let result = refinement_span.in_scope(|| {
            let mut attempts = 0;
            let mut last_issues = vec![];

            for attempt in 1..=self.max_attempts {
                attempts = attempt;

                let attempt_span = info_span!("refinement_attempt", attempt = attempt);
                let attempt_result = attempt_span.in_scope(|| {
                    // Validation + issue analysis
                    let validator = HtmlValidator::new();
                    let issues = validator.validate(""); // placeholder

                    if issues.is_empty() {
                        return Some(AdvancedOrchestrationResult {
                            success: true,
                            attempts_used: attempts,
                            ..Default::default()
                        });
                    }

                    warn!(issues_count = issues.len(), "Refinement needed");
                    last_issues = issues;
                    None
                });

                if let Some(success) = attempt_result {
                    return success;
                }
            }

            AdvancedOrchestrationResult {
                success: false,
                attempts_used: attempts,
                validation_issues: last_issues,
                ..Default::default()
            }
        });

        let duration = start.elapsed().as_secs_f64();
        observability::record_orchestration_metrics(duration, result.success, result.attempts_used);

        result
    }
}

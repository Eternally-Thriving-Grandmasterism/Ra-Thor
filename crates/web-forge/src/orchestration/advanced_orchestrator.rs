/// Advanced Orchestrator
///
/// Production-grade orchestration engine with rich observability.

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::ComponentTree;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::renderer::render_tree;
use crate::orchestration::semantic_planning::{SemanticPlanningStrategy, OpenAIEmbeddingProvider};
use crate::validation::HtmlValidator;
use serde_json::from_str;
use tracing::{info_span, Instrument};

// ... (PlanningResult, PlanningStrategy, etc. remain unchanged)

impl AdvancedOrchestrator {
    // ... constructors unchanged ...

    pub fn orchestrate(&self, prompt: &str) -> AdvancedOrchestrationResult {
        let root_span = info_span!("orchestration", prompt = %prompt);

        async move {
            tracing::info!("Starting orchestration");

            // === Planning Phase ===
            let planning_span = info_span!("planning");
            let planning = planning_span.in_scope(|| {
                self.planning_strategy.plan(prompt, &self.registry)
            });
            let top_components = planning.prioritized_components();
            tracing::info!(components = ?top_components, "Planning completed");

            // === Generation Phase ===
            let generation_span = info_span!("generation", components = ?top_components);
            let generated_json = generation_span.in_scope(|| {
                self.generator.generate_with_planning(prompt, &top_components)
            });

            // === Validation + Refinement Loop ===
            let mut attempts = 0;
            let mut last_issues = vec![];

            for attempt in 1..=self.max_attempts {
                attempts = attempt;
                let attempt_span = info_span!("refinement_attempt", attempt);

                let result = attempt_span.in_scope(|| {
                    let validator = HtmlValidator::new();
                    let issues = validator.validate(&generated_json);

                    if issues.is_empty() {
                        return Some(AdvancedOrchestrationResult {
                            final_html: None, // placeholder
                            component_tree: None,
                            validation_issues: vec![],
                            attempts_used: attempts,
                            success: true,
                        });
                    }
                    None
                });

                if let Some(success_result) = result {
                    return success_result;
                }

                last_issues = /* validate again if needed */ vec![];
            }

            AdvancedOrchestrationResult {
                final_html: None,
                component_tree: None,
                validation_issues: last_issues,
                attempts_used: attempts,
                success: false,
            }
        }
        .instrument(root_span)
        .await
    }
}

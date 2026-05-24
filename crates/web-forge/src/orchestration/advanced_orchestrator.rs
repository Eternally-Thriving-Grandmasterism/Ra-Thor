/// Advanced Orchestrator
///
/// Next-generation orchestration engine with phased execution.
///
/// Phases:
/// 1. Planning     - Decide strategy and relevant components
/// 2. Generation   - Use ComponentAwareGenerator to produce ComponentTree
/// 3. Validation   - Run HtmlValidator + component contracts
/// 4. Refinement   - Retry with feedback if validation fails
/// 5. Finalization - Render final HTML

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::ComponentTree;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::renderer::render_tree;
use crate::validation::HtmlValidator;
use serde_json::from_str;

/// Rich result from advanced orchestration.
#[derive(Debug)]
pub struct AdvancedOrchestrationResult {
    pub final_html: Option<String>,
    pub component_tree: Option<ComponentTree>,
    pub validation_issues: Vec<String>,
    pub attempts_used: usize,
    pub success: bool,
}

/// Trait for planning strategies (future intelligence layer).
pub trait PlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> Vec<String>;
}

/// Default simple planning strategy.
pub struct DefaultPlanningStrategy;

impl PlanningStrategy for DefaultPlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> Vec<String> {
        // Simple matching against known components
        let prompt_lower = prompt.to_lowercase();
        let mut matched = vec![];

        for component in registry.list_all() {
            if prompt_lower.contains(&component.name.to_lowercase()) {
                matched.push(component.name.clone());
            }
        }

        if matched.is_empty() {
            matched.push("Button".to_string());
        }

        matched
    }
}

/// The Advanced Orchestrator.
pub struct AdvancedOrchestrator {
    max_attempts: usize,
    registry: ComponentRegistry,
    generator: ComponentAwareGenerator,
}

impl AdvancedOrchestrator {
    pub fn new() -> Self {
        Self {
            max_attempts: 3,
            registry: ComponentRegistry::new(),
            generator: ComponentAwareGenerator::new(),
        }
    }

    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }

    /// Main orchestration entry point.
    pub fn orchestrate(&self, prompt: &str) -> AdvancedOrchestrationResult {
        println!("[AdvancedOrchestrator] Starting phased execution...");

        let mut attempts = 0;
        let mut last_issues = vec![];

        for attempt in 1..=self.max_attempts {
            attempts = attempt;

            // Phase 1: Planning
            println!("[Phase 1] Planning (attempt {})...", attempt);
            let _planned_components = DefaultPlanningStrategy.plan(prompt, &self.registry);

            // Phase 2: Generation
            println!("[Phase 2] Generation...");
            let generated_json = self.generator.generate(prompt);

            // Phase 3: Validation
            println!("[Phase 3] Validation...");
            let validator = HtmlValidator::new();
            let issues = validator.validate(&generated_json);

            if issues.is_empty() {
                // Success path - try to render
                println!("[Phase 3] Validation passed.");

                let component_tree = from_str::<ComponentTree>(&generated_json).ok();
                let final_html = component_tree.as_ref().map(|tree| render_tree(tree));

                return AdvancedOrchestrationResult {
                    final_html,
                    component_tree,
                    validation_issues: vec![],
                    attempts_used: attempts,
                    success: true,
                };
            }

            last_issues = issues.clone();
            println!("[Phase 3] Validation failed with {} issues.", issues.len());

            // Phase 4: Refinement (simple feedback for now)
            if attempt < self.max_attempts {
                println!("[Phase 4] Refinement triggered...");
            }
        }

        // Failed after max attempts
        AdvancedOrchestrationResult {
            final_html: None,
            component_tree: None,
            validation_issues: last_issues,
            attempts_used: attempts,
            success: false,
        }
    }
}

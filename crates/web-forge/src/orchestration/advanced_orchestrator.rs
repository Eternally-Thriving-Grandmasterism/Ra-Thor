/// Advanced Orchestrator
///
/// Enhanced with richer Planning phase and PlanningStrategy.

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::ComponentTree;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::renderer::render_tree;
use crate::validation::HtmlValidator;
use serde_json::from_str;

#[derive(Debug)]
pub struct AdvancedOrchestrationResult {
    pub final_html: Option<String>,
    pub component_tree: Option<ComponentTree>,
    pub validation_issues: Vec<String>,
    pub attempts_used: usize,
    pub success: bool,
}

/// Rich output from the Planning phase.
#[derive(Debug, Clone)]
pub struct PlanningResult {
    pub intent: String,
    pub suggested_components: Vec<String>,
    pub constraints: Vec<String>,
    pub confidence: f32,
}

/// Trait for planning strategies.
pub trait PlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult;
}

/// Default planning strategy with basic intelligence.
pub struct DefaultPlanningStrategy;

impl PlanningStrategy for DefaultPlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult {
        let prompt_lower = prompt.to_lowercase();
        let mut suggested = vec![];

        for component in registry.list_all() {
            if prompt_lower.contains(&component.name.to_lowercase()) {
                suggested.push(component.name.clone());
            }
        }

        if suggested.is_empty() {
            suggested.push("Button".to_string());
        }

        PlanningResult {
            intent: prompt.to_string(),
            suggested_components: suggested,
            constraints: vec![],
            confidence: if suggested.len() > 1 { 0.8 } else { 0.6 },
        }
    }
}

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

    pub fn orchestrate(&self, prompt: &str) -> AdvancedOrchestrationResult {
        println!("[AdvancedOrchestrator] Starting phased execution...");

        let mut attempts = 0;
        let mut last_issues = vec![];

        for attempt in 1..=self.max_attempts {
            attempts = attempt;

            // === Phase 1: Planning (now richer) ===
            println!("[Phase 1] Planning (attempt {})...", attempt);
            let planning_result = DefaultPlanningStrategy.plan(prompt, &self.registry);
            println!("[Planning] Suggested components: {:?} (confidence: {:.1})",
                     planning_result.suggested_components, planning_result.confidence);

            // === Phase 2: Generation ===
            println!("[Phase 2] Generation...");
            let generated_json = self.generator.generate(prompt);

            // === Phase 3: Validation ===
            println!("[Phase 3] Validation...");
            let validator = HtmlValidator::new();
            let issues = validator.validate(&generated_json);

            if issues.is_empty() {
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
            println!("[Phase 3] Validation failed ({} issues).", issues.len());

            if attempt < self.max_attempts {
                println!("[Phase 4] Refinement triggered...");
            }
        }

        AdvancedOrchestrationResult {
            final_html: None,
            component_tree: None,
            validation_issues: last_issues,
            attempts_used: attempts,
            success: false,
        }
    }
}

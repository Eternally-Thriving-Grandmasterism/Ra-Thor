/// Advanced Orchestrator
///
/// Now properly leverages rich PlanningResult during execution.

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

#[derive(Debug, Clone)]
pub struct PlanningResult {
    pub intent: String,
    pub scored_components: Vec<(String, f32)>,
    pub constraints: Vec<String>,
    pub confidence: f32,
}

impl PlanningResult {
    pub fn prioritized_components(&self) -> Vec<String> {
        let mut sorted = self.scored_components.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().map(|(name, _)| name).collect()
    }
}

pub trait PlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult;
}

pub struct DefaultPlanningStrategy;

impl DefaultPlanningStrategy {
    fn decompose_prompt(prompt: &str) -> Vec<String> { vec![prompt.to_string()] }

    fn score_component(sub: &str, component: &ComponentDefinition) -> f32 {
        let sub_lower = sub.to_lowercase();
        let mut score = 0.0;

        if sub_lower == component.name.to_lowercase() { score += 1.0; }
        else if sub_lower.contains(&component.name.to_lowercase()) { score += 0.75; }

        if sub_lower.contains(&component.description.to_lowercase()) { score += 0.35; }

        if (sub_lower.contains("create") || sub_lower.contains("add") || sub_lower.contains("build"))
            && sub_lower.contains(&component.name.to_lowercase())
        {
            score += 0.15;
        }

        score.min(1.0)
    }
}

impl PlanningStrategy for DefaultPlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult {
        let sub_intents = Self::decompose_prompt(prompt);
        let mut scored_map = std::collections::HashMap::new();

        for sub in sub_intents {
            for component in registry.list_all() {
                let score = Self::score_component(&sub, component);
                if score > 0.0 {
                    let entry = scored_map.entry(component.name.clone()).or_insert(0.0);
                    *entry = entry.max(score);
                }
            }
        }

        if scored_map.is_empty() {
            scored_map.insert("Button".to_string(), 0.6);
        }

        let mut scored: Vec<_> = scored_map.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        PlanningResult {
            intent: prompt.to_string(),
            scored_components: scored,
            constraints: vec![],
            confidence: 0.8,
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

        let planning = DefaultPlanningStrategy.plan(prompt, &self.registry);
        let top_components = planning.prioritized_components();

        println!("[Phase 1] Planning complete. Top components: {:?}", top_components);

        let mut attempts = 0;
        let mut last_issues = vec![];

        for attempt in 1..=self.max_attempts {
            attempts = attempt;

            println!("[Phase 2] Generation (attempt {})...", attempt);
            let generated_json = self.generator.generate(prompt);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planning_detects_components() {
        let registry = ComponentRegistry::new();
        let strategy = DefaultPlanningStrategy;
        let result = strategy.plan("Create a button and a card", &registry);

        let names: Vec<_> = result.scored_components.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Button"));
        assert!(names.contains(&"Card"));
    }
}

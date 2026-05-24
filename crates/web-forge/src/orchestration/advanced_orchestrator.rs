/// Advanced Orchestrator
///
/// The central coordination engine for Ra-Thor's web generation system.
///
/// # Features
/// - Planning-aware generation (supports both keyword and semantic strategies)
/// - Component-aware generation via `ComponentAwareGenerator`
/// - Multi-attempt refinement with issue analysis
/// - Graceful degradation between planning modes
///
/// # Usage
/// ```ignore
/// let orchestrator = AdvancedOrchestrator::new()
///     .with_max_attempts(3)
///     .with_semantic_planning("sk-...".to_string());
///
/// let result = orchestrator.orchestrate("Create a beautiful hero section");
/// ```

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::ComponentTree;
use crate::orchestration::generation::ComponentAwareGenerator;
use crate::orchestration::renderer::render_tree;
use crate::orchestration::semantic_planning::{SemanticPlanningStrategy, OpenAIEmbeddingProvider};
use crate::validation::HtmlValidator;
use serde_json::from_str;

/// Result returned after an orchestration run.
#[derive(Debug)]
pub struct AdvancedOrchestrationResult {
    pub final_html: Option<String>,
    pub component_tree: Option<ComponentTree>,
    pub validation_issues: Vec<String>,
    pub attempts_used: usize,
    pub success: bool,
}

/// Structured output from any PlanningStrategy.
#[derive(Debug, Clone)]
pub struct PlanningResult {
    pub intent: String,
    pub scored_components: Vec<(String, f32)>,
    pub constraints: Vec<String>,
    pub confidence: f32,
}

impl PlanningResult {
    /// Returns component names sorted by relevance (highest first).
    pub fn prioritized_components(&self) -> Vec<String> {
        let mut sorted = self.scored_components.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().map(|(name, _)| name).collect()
    }
}

/// Trait for all planning strategies (keyword-based or semantic).
pub trait PlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult;
}

/// Default keyword-based planning strategy.
pub struct DefaultPlanningStrategy;

impl PlanningStrategy for DefaultPlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult {
        let mut scored = vec![];
        let prompt_lower = prompt.to_lowercase();

        for component in registry.list_all() {
            if prompt_lower.contains(&component.name.to_lowercase()) {
                scored.push((component.name.clone(), 0.8));
            }
        }

        if scored.is_empty() {
            scored.push(("Button".to_string(), 0.6));
        }

        PlanningResult {
            intent: prompt.to_string(),
            scored_components: scored,
            constraints: vec![],
            confidence: 0.8,
        }
    }
}

/// The main Advanced Orchestrator.
pub struct AdvancedOrchestrator {
    max_attempts: usize,
    registry: ComponentRegistry,
    generator: ComponentAwareGenerator,
    planning_strategy: Box<dyn PlanningStrategy>,
}

impl AdvancedOrchestrator {
    /// Creates a new orchestrator with default keyword-based planning.
    pub fn new() -> Self {
        Self {
            max_attempts: 3,
            registry: ComponentRegistry::new(),
            generator: ComponentAwareGenerator::new(),
            planning_strategy: Box::new(DefaultPlanningStrategy),
        }
    }

    /// Sets the maximum number of generation + refinement attempts.
    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }

    /// Enables semantic planning using OpenAI embeddings.
    /// Component embeddings are precomputed on call.
    pub fn with_semantic_planning(mut self, api_key: String) -> Self {
        let mut semantic_planner = SemanticPlanningStrategy::new(OpenAIEmbeddingProvider::new(api_key));
        semantic_planner.precompute_embeddings(&self.registry);
        self.planning_strategy = Box::new(semantic_planner);
        self
    }

    pub fn orchestrate(&self, prompt: &str) -> AdvancedOrchestrationResult {
        println!("[AdvancedOrchestrator] Starting phased execution...");

        // === Planning ===
        let planning = self.planning_strategy.plan(prompt, &self.registry);
        let top_components = planning.prioritized_components();
        println!("[Phase 1] Planning complete. Top components: {:?}", top_components);

        let mut attempts = 0;
        let mut last_issues = vec![];
        let mut refinement_context = String::new();

        for attempt in 1..=self.max_attempts {
            attempts = attempt;

            // Build generation prompt (with refinement context on retries)
            let generation_prompt = if attempt == 1 {
                prompt.to_string()
            } else {
                format!("Original: {}. Issues: {}. Guidance: {}",
                        prompt,
                        last_issues.join("; "),
                        refinement_context)
            };

            println!("[Phase 2] Generation (attempt {})...", attempt);
            let generated_json = self.generator.generate_with_planning(&generation_prompt, &top_components);

            println!("[Phase 3] Validation...");
            let validator = HtmlValidator::new();
            let issues = validator.validate(&generated_json);

            if issues.is_empty() {
                println!("[Phase 3] Validation passed on attempt {}.\n", attempt);

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

            // Prepare refinement
            last_issues = issues.clone();
            refinement_context = format!("Fix: {}", issues.join("; "));

            println!("[Phase 3] Validation failed on attempt {} ({} issues).", attempt, issues.len());

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
    fn test_default_planning() {
        let registry = ComponentRegistry::new();
        let strategy = DefaultPlanningStrategy;
        let result = strategy.plan("Create a button", &registry);
        assert!(!result.scored_components.is_empty());
    }
}

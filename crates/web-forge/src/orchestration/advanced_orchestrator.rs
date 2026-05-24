/// Advanced Orchestrator
///
/// Enhanced Planning phase with relevance scoring and prioritization.

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

/// Rich structured output from Planning.
#[derive(Debug, Clone)]
pub struct PlanningResult {
    pub intent: String,
    /// Components with relevance scores (name, score)
    pub scored_components: Vec<(String, f32)>,
    pub constraints: Vec<String>,
    pub confidence: f32,
}

impl PlanningResult {
    /// Returns components sorted by relevance (highest first)
    pub fn prioritized_components(&self) -> Vec<String> {
        let mut sorted = self.scored_components.clone();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().map(|(name, _)| name).collect()
    }
}

/// Trait for planning strategies.
pub trait PlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult;
}

/// Default planning strategy with relevance scoring.
pub struct DefaultPlanningStrategy;

impl PlanningStrategy for DefaultPlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult {
        let prompt_lower = prompt.to_lowercase();
        let mut scored = vec![];

        for component in registry.list_all() {
            let name_lower = component.name.to_lowercase();
            let mut score: f32 = 0.0;

            // Simple relevance scoring
            if prompt_lower.contains(&name_lower) {
                score += 0.7;
            }
            if prompt_lower.contains(&component.description.to_lowercase()) {
                score += 0.3;
            }

            if score > 0.0 {
                scored.push((component.name.clone(), score.min(1.0)));
            }
        }

        // Fallback
        if scored.is_empty() {
            scored.push(("Button".to_string(), 0.5));
        }

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        PlanningResult {
            intent: prompt.to_string(),
            scored_components: scored.clone(),
            constraints: vec![],
            confidence: if scored.len() > 1 { 0.85 } else { 0.65 },
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

            // === Phase 1: Planning with scoring ===
            println!("[Phase 1] Planning (attempt {})...", attempt);
            let planning = DefaultPlanningStrategy.plan(prompt, &self.registry);

            println!("[Planning] Prioritized components:");
            for (name, score) in &planning.scored_components {
                println!("   - {} (relevance: {:.2})", name, score);
            }

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

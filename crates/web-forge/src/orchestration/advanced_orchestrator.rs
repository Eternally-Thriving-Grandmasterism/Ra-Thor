/// Advanced Orchestrator
///
/// Hybrid Multi-Intent Planning with Optimized Relevance Scoring.

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

/// Hybrid Multi-Intent Planning Strategy with Optimized Scoring
pub struct DefaultPlanningStrategy;

impl DefaultPlanningStrategy {
    fn decompose_prompt(prompt: &str) -> Vec<String> {
        let separators = [" and ", " with ", " plus ", ", ", " & "];
        let mut parts = vec![prompt.to_string()];

        for sep in separators {
            let mut new_parts = vec![];
            for part in parts {
                for sub in part.split(sep) {
                    let trimmed = sub.trim().to_string();
                    if !trimmed.is_empty() {
                        new_parts.push(trimmed);
                    }
                }
            }
            parts = new_parts;
        }
        parts
    }
}

impl PlanningStrategy for DefaultPlanningStrategy {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult {
        let sub_intents = Self::decompose_prompt(prompt);
        let mut scored_map: std::collections::HashMap<String, f32> = std::collections::HashMap::new();

        for sub in sub_intents {
            let sub_lower = sub.to_lowercase();

            for component in registry.list_all() {
                let name_lower = component.name.to_lowercase();
                let mut score: f32 = 0.0;

                // Optimized Relevance Scoring
                if sub_lower == name_lower {
                    score += 1.0; // Exact match (very strong)
                } else if sub_lower.contains(&name_lower) {
                    score += 0.75; // Strong partial match
                }

                // Description match (weaker signal)
                if sub_lower.contains(&component.description.to_lowercase()) {
                    score += 0.35;
                }

                // Action word boost
                if sub_lower.contains("create") || sub_lower.contains("add") || sub_lower.contains("build") {
                    if sub_lower.contains(&name_lower) {
                        score += 0.15;
                    }
                }

                if score > 0.0 {
                    let entry = scored_map.entry(component.name.clone()).or_insert(0.0);
                    *entry = (*entry + score).min(1.0);
                }
            }
        }

        if scored_map.is_empty() {
            scored_map.insert("Button".to_string(), 0.6);
        }

        let mut scored: Vec<(String, f32)> = scored_map.into_iter().collect();
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

        let mut attempts = 0;
        let mut last_issues = vec![];

        for attempt in 1..=self.max_attempts {
            attempts = attempt;

            println!("[Phase 1] Planning (attempt {})...", attempt);
            let planning = DefaultPlanningStrategy.plan(prompt, &self.registry);

            println!("[Planning] Prioritized components:");
            for (name, score) in &planning.scored_components {
                println!("   - {} (relevance: {:.2})", name, score);
            }

            println!("[Phase 2] Generation...");
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

/// Generation Module
///
/// Planning-aware component generator.

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::{ComponentNode, ComponentTree};
use crate::orchestration::GenerationStrategy;
use serde_json::json;

pub struct ComponentAwareGenerator {
    registry: ComponentRegistry,
}

impl ComponentAwareGenerator {
    pub fn new() -> Self {
        Self {
            registry: ComponentRegistry::new(),
        }
    }

    /// Generate while preferring components suggested by planning.
    pub fn generate_with_planning(&self, prompt: &str, planned_components: &[String]) -> String {
        let mut components_to_use = vec![];

        // Prefer components from planning
        for name in planned_components {
            if self.registry.get(name).is_some() {
                components_to_use.push(name.clone());
            }
        }

        // Fallback to prompt-based mapping if planning gave nothing useful
        if components_to_use.is_empty() {
            components_to_use = self.map_to_components(prompt);
        }

        let mut children = vec![];
        for comp_name in components_to_use {
            if let Some(def) = self.registry.get(&comp_name) {
                let node = ComponentNode {
                    component: comp_name,
                    props: json!({ "variant": "Primary" }),
                    children: vec![],
                    text: Some(format!("Generated {}", def.name)),
                };
                children.push(node);
            }
        }

        let root = ComponentNode {
            component: "div".to_string(),
            props: json!({ "class": "generated-content" }),
            children,
            text: None,
        };

        let tree = ComponentTree::new(root);
        serde_json::to_string_pretty(&tree).unwrap_or_default()
    }

    fn map_to_components(&self, prompt: &str) -> Vec<String> {
        let prompt_lower = prompt.to_lowercase();
        let mut matched = vec![];

        for component in self.registry.list_all() {
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

impl GenerationStrategy for ComponentAwareGenerator {
    fn generate(&self, prompt: &str) -> String {
        self.generate_with_planning(prompt, &[])
    }
}

/// Generation Module
///
/// Production-grade component-aware generator.
///
/// This generator prefers known components from the ComponentRegistry
/// and falls back intelligently when encountering unknown concepts.

use crate::orchestration::component_registry::ComponentRegistry;
use crate::orchestration::component_tree::{ComponentNode, ComponentTree};
use crate::orchestration::GenerationStrategy;
use serde_json::json;

/// A component-aware generator that targets the ComponentTree format.
pub struct ComponentAwareGenerator {
    registry: ComponentRegistry,
}

impl ComponentAwareGenerator {
    pub fn new() -> Self {
        Self {
            registry: ComponentRegistry::new(),
        }
    }

    /// Attempts to map a prompt to known components.
    fn map_to_components(&self, prompt: &str) -> Vec<String> {
        let prompt_lower = prompt.to_lowercase();
        let mut matched = vec![];

        for component in self.registry.list_all() {
            if prompt_lower.contains(&component.name.to_lowercase()) {
                matched.push(component.name.clone());
            }
        }

        // Intelligent fallback
        if matched.is_empty() {
            matched.push("Button".to_string());
        }

        matched
    }
}

impl GenerationStrategy for ComponentAwareGenerator {
    fn generate(&self, prompt: &str) -> String {
        let matched_components = self.map_to_components(prompt);

        let mut children = vec![];

        for comp_name in matched_components {
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
}

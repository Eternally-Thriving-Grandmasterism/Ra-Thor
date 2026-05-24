/// Component Registry
///
/// A rich registry that describes components in detail for AI generation.
/// This enables component-aware, production-grade generation.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ComponentProp {
    pub name: String,
    pub prop_type: String,
    pub required: bool,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ComponentDefinition {
    pub name: String,
    pub description: String,
    pub category: String,
    pub props: Vec<ComponentProp>,
    pub requires_token_compliance: bool,
    pub example_usage: String,
}

pub struct ComponentRegistry {
    components: HashMap<String, ComponentDefinition>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            components: HashMap::new(),
        };
        registry.register_core_components();
        registry
    }

    fn register_core_components(&mut self) {
        self.register(ComponentDefinition {
            name: "Button".to_string(),
            description: "A clickable button component with variants and sizes.".to_string(),
            category: "action".to_string(),
            props: vec![
                ComponentProp { name: "variant".to_string(), prop_type: "Primary | Secondary".to_string(), required: true, description: "Visual style".to_string() },
                ComponentProp { name: "size".to_string(), prop_type: "Small | Medium | Large".to_string(), required: false, description: "Size".to_string() },
            ],
            requires_token_compliance: true,
            example_usage: "<button class=\"btn btn-Primary\">Click</button>".to_string(),
        });

        self.register(ComponentDefinition {
            name: "Card".to_string(),
            description: "A container for grouping content.".to_string(),
            category: "layout".to_string(),
            props: vec![ ComponentProp { name: "padding".to_string(), prop_type: "Small | Medium | Large".to_string(), required: false, description: "Padding".to_string() } ],
            requires_token_compliance: true,
            example_usage: "<div class=\"card\">...</div>".to_string(),
        });
    }

    pub fn register(&mut self, definition: ComponentDefinition) {
        self.components.insert(definition.name.clone(), definition);
    }

    pub fn get(&self, name: &str) -> Option<&ComponentDefinition> {
        self.components.get(name)
    }

    pub fn list_all(&self) -> Vec<&ComponentDefinition> {
        self.components.values().collect()
    }

    pub fn count(&self) -> usize {
        self.components.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_contains_button_and_card() {
        let registry = ComponentRegistry::new();
        assert!(registry.get("Button").is_some());
        assert!(registry.get("Card").is_some());
    }

    #[test]
    fn test_registry_count() {
        let registry = ComponentRegistry::new();
        assert!(registry.count() >= 2);
    }

    #[test]
    fn test_list_all_returns_components() {
        let registry = ComponentRegistry::new();
        let all = registry.list_all();
        assert!(!all.is_empty());
    }

    #[test]
    fn test_get_nonexistent_returns_none() {
        let registry = ComponentRegistry::new();
        assert!(registry.get("NonExistentComponent").is_none());
    }
}

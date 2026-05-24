/// Component Registry
///
/// A rich registry that describes components in detail for AI generation.
/// This enables component-aware, production-grade generation.

use std::collections::HashMap;

/// Describes a single prop of a component.
#[derive(Debug, Clone)]
pub struct ComponentProp {
    pub name: String,
    pub prop_type: String,
    pub required: bool,
    pub description: String,
}

/// Rich definition of a component for generation and validation.
#[derive(Debug, Clone)]
pub struct ComponentDefinition {
    pub name: String,
    pub description: String,
    pub category: String,
    pub props: Vec<ComponentProp>,
    pub requires_token_compliance: bool,
    pub example_usage: String,
}

/// Central registry of all available components.
pub struct ComponentRegistry {
    components: HashMap<String, ComponentDefinition>,
}

impl ComponentRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            components: HashMap::new(),
        };

        // Register core components
        registry.register_core_components();
        registry
    }

    fn register_core_components(&mut self) {
        // Button
        self.register(ComponentDefinition {
            name: "Button".to_string(),
            description: "A clickable button component with variants and sizes.".to_string(),
            category: "action".to_string(),
            props: vec![
                ComponentProp {
                    name: "variant".to_string(),
                    prop_type: "Primary | Secondary".to_string(),
                    required: true,
                    description: "Visual style of the button".to_string(),
                },
                ComponentProp {
                    name: "size".to_string(),
                    prop_type: "Small | Medium | Large".to_string(),
                    required: false,
                    description: "Size of the button".to_string(),
                },
            ],
            requires_token_compliance: true,
            example_usage: "<button class=\"btn btn-Primary\">Click me</button>".to_string(),
        });

        // Card
        self.register(ComponentDefinition {
            name: "Card".to_string(),
            description: "A container component for grouping related content.".to_string(),
            category: "layout".to_string(),
            props: vec![
                ComponentProp {
                    name: "padding".to_string(),
                    prop_type: "Small | Medium | Large".to_string(),
                    required: false,
                    description: "Internal spacing of the card".to_string(),
                },
            ],
            requires_token_compliance: true,
            example_usage: "<div class=\"card card-Medium\">...</div>".to_string(),
        });

        // TODO: Add Input, Modal, and future components
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

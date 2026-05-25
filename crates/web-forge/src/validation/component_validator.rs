/// Component-Aware Validation
///
/// Truth: For AI generation to be reliable, the Validation Engine
/// must understand the components it is validating.
///
/// This module begins the deep integration between the Validation Engine
/// and the Component System.

use std::collections::HashSet;

pub struct ComponentValidator {
    known_components: HashSet<String>,
}

impl ComponentValidator {
    pub fn new() -> Self {
        Self {
            known_components: HashSet::new(),
        }
    }

    /// Register a component that the validator should recognize
    pub fn register_component(&mut self, name: &str) {
        self.known_components.insert(name.to_string());
    }

    /// Check if a component name is known/registered
    pub fn is_known_component(&self, name: &str) -> bool {
        self.known_components.contains(name)
    }

    /// Validate that all used components are known
    /// This is a foundational integration step.
    pub fn validate_known_components(&self, used_components: &[String]) -> Vec<String> {
        let mut issues = vec![];

        for component in used_components {
            if !self.is_known_component(component) {
                issues.push(format!("Unknown component used: '{}'", component));
            }
        }

        issues
    }
}
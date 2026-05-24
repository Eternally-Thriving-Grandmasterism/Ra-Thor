/// HTML Validator with ComponentValidator integration
use crate::validation::component_validator::ComponentValidator;
use crate::validation::rules;

pub struct HtmlValidator {
    component_validator: ComponentValidator,
    strict_mode: bool,
}

impl HtmlValidator {
    pub fn new() -> Self {
        let mut validator = Self {
            component_validator: ComponentValidator::new(),
            strict_mode: false,
        };

        validator.register_component("Button");
        validator.register_component("Card");
        validator.register_component("Input");
        validator.register_component("Modal");

        validator
    }

    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    pub fn register_component(&mut self, name: &str) {
        self.component_validator.register_component(name);
    }

    pub fn validate(&self, html: &str) -> Vec<String> {
        let mut issues = Vec::new();

        issues.extend(rules::no_markdown_artifacts::check(html));
        issues.extend(rules::proper_details::check(html));
        issues.extend(rules::language_switcher_ids::check(html));
        issues.extend(rules::required_ids::check(html));
        issues.extend(rules::accessibility_basic::check(html));
        issues.extend(rules::token_compliance::check(html));

        if self.strict_mode && !issues.is_empty() {
            issues.push("Strict mode: All issues must be resolved.".to_string());
        }

        issues
    }

    pub fn is_valid(&self, html: &str) -> bool {
        self.validate(html).is_empty()
    }

    /// Placeholder for future component-specific validation
    pub fn validate_component(&self, component_name: &str, html_fragment: &str) -> Vec<String> {
        // In a full implementation, we would look up the component
        // and call its custom validate() method.
        let mut issues = vec![];

        if !self.component_validator.is_known_component(component_name) {
            issues.push(format!("Unknown component: {}", component_name));
        }

        issues
    }
}
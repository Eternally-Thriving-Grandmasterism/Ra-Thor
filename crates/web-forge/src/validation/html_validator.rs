/// HTML Validator with ComponentValidator integration
///
/// By default, validate() now sanitizes input first for safety.
use crate::validation::component_validator::ComponentValidator;
use crate::validation::rules;
use crate::validation::validate::Validate;
use crate::component_system::contract::ComponentContract;
use crate::sanitizer;

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

    /// Main validation method.
    /// Automatically sanitizes input first (recommended default behavior).
    pub fn validate(&self, html: &str) -> Vec<String> {
        let clean_html = sanitizer::sanitize(html);
        let mut issues = Vec::new();

        issues.extend(rules::no_markdown_artifacts::check(&clean_html));
        issues.extend(rules::proper_details::check(&clean_html));
        issues.extend(rules::language_switcher_ids::check(&clean_html));
        issues.extend(rules::required_ids::check(&clean_html));
        issues.extend(rules::accessibility_basic::check(&clean_html));
        issues.extend(rules::token_compliance::check(&clean_html));

        if self.strict_mode && !issues.is_empty() {
            issues.push("Strict mode: All issues must be resolved.".to_string());
        }

        issues
    }

    pub fn is_valid(&self, html: &str) -> bool {
        self.validate(html).is_empty()
    }

    /// Explicit sanitize + validate (returns cleaned HTML too)
    pub fn sanitize_and_validate(&self, html: &str) -> (String, Vec<String>) {
        let clean_html = sanitizer::sanitize(html);
        let issues = self.validate(&clean_html); // already sanitized, but safe to call
        (clean_html, issues)
    }

    pub fn validate_with_component<T: ComponentContract>(&self, component: &T, html_fragment: &str) -> Vec<String> {
        let mut issues = vec![];

        if !self.component_validator.is_known_component(component.name()) {
            issues.push(format!("Component '{}' is not registered", component.name()));
        }

        issues.extend(component.validate());

        issues
    }
}
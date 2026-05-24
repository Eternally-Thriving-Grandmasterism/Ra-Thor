/// HTML Validator
///
/// Now includes structured WCAG AA validation logic.

use crate::validation::rules;
use crate::sanitizer;
use crate::html_parser;

pub struct HtmlValidator {
    strict_mode: bool,
}

impl HtmlValidator {
    pub fn new() -> Self {
        Self { strict_mode: false }
    }

    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    pub fn validate(&self, html: &str) -> Vec<String> {
        let clean_html = sanitizer::sanitize(html);
        let mut issues = Vec::new();

        // Existing landmark checks...
        if !html_parser::has_element(&clean_html, "main") {
            issues.push("Missing main landmark".to_string());
        }

        // Include structured WCAG AA checks
        issues.extend(rules::wcag_aa::check(&clean_html));

        // Keep other existing rules
        issues.extend(rules::accessibility_basic::check(&clean_html));

        if self.strict_mode && !issues.is_empty() {
            issues.push("Strict mode active: All issues must be resolved.".to_string());
        }

        issues
    }

    pub fn is_valid(&self, html: &str) -> bool {
        self.validate(html).is_empty()
    }
}

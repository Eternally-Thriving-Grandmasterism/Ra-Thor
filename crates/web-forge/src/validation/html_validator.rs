/// HTML Validator
///
/// Provides structural validation for generated HTML.
/// Part of the Cathedral approach in web-forge.

pub struct HtmlValidator;

impl HtmlValidator {
    pub fn new() -> Self {
        Self
    }

    /// Basic validation placeholder
    pub fn validate(&self, html: &str) -> Vec<String> {
        let mut issues = vec![];

        if html.contains("**Summary:**") {
            issues.push("Found leftover markdown '**Summary:**' artifacts".to_string());
        }

        if !html.contains("<details") && html.contains("<summary>") {
            issues.push("Found <summary> without wrapping <details>".to_string());
        }

        issues
    }
}
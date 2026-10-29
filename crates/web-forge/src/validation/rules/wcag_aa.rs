/// WCAG 2.1 AA Validation Logic
///
/// Structured checks against key WCAG AA success criteria.
/// This module provides more rigorous validation than basic accessibility checks.

pub fn check(html: &str) -> Vec<String> {
    let mut issues = Vec::new();

    // === 1.1.1 Non-text Content (Level A) ===
    if html.contains("<img") && !html.contains("alt=") {
        issues.push("[1.1.1] Images must have alt text (Non-text Content)".to_string());
    }

    // === 1.3.1 Info and Relationships (Level A) ===
    if html.contains("<input") && !html.contains("<label") {
        issues.push("[1.3.1] Form inputs should have associated labels (Info and Relationships)".to_string());
    }

    // === 2.4.1 Bypass Blocks (Level A) ===
    if !html.contains("<main") && !html.contains("role=\"main\"") {
        issues.push("[2.4.1] Page should provide a way to bypass repeated content (e.g. main landmark)".to_string());
    }

    // === 2.4.3 Focus Order (Level A) ===
    if html.contains("onclick=") && !html.contains("tabindex=") {
        issues.push("[2.4.3] Interactive elements should be keyboard accessible (Focus Order)".to_string());
    }

    // === 3.3.2 Labels or Instructions (Level A) ===
    if html.contains("<form") && !html.contains("<label") {
        issues.push("[3.3.2] Forms should provide labels or instructions for inputs".to_string());
    }

    // === 4.1.2 Name, Role, Value (Level A) ===
    if html.contains("<button") && !html.contains("> ") && !html.contains("aria-label") {
        issues.push("[4.1.2] Buttons must have accessible names (Name, Role, Value)".to_string());
    }

    issues
}

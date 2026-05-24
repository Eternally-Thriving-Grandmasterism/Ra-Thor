/// Accessibility Compliance Checks (WCAG-aligned)
///
/// These checks help ensure generated HTML meets basic accessibility standards.

pub fn check(html: &str) -> Vec<String> {
    let mut issues = Vec::new();

    // Images must have alt text
    if html.contains("<img") && !html.contains("alt=") {
        issues.push("Images must have alt attributes for accessibility".to_string());
    }

    // Form controls should have labels
    if html.contains("<input") && !html.contains("<label") {
        issues.push("Form inputs should have associated <label> elements".to_string());
    }

    // Buttons should have accessible names
    if html.contains("<button") {
        // Basic heuristic — real parsing would be better
        if !html.contains("> ") && !html.contains("aria-label") {
            issues.push("Buttons should have visible text or aria-label".to_string());
        }
    }

    // Encourage proper heading structure
    if !html.contains("<h1") {
        issues.push("Page should contain at least one <h1> heading".to_string());
    }

    // Interactive elements should support keyboard navigation
    if html.contains("onclick=") && !html.contains("tabindex=") {
        issues.push("Elements with onclick should support keyboard interaction (tabindex)".to_string());
    }

    issues
}

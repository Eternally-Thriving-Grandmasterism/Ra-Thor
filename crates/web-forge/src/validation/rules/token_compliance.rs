/// Basic Design Token Compliance Rule
///
/// Truth: Generated UI should respect the design token system
/// to maintain visual consistency and theming integrity.

pub fn check(html: &str) -> Vec<String> {
    let mut issues = vec![];

    // Simple heuristic: Check if common token classes are used
    if html.contains("class=\"btn\"") && !html.contains("btn-") {
        issues.push("Button found without token-based variant classes".to_string());
    }

    issues
}
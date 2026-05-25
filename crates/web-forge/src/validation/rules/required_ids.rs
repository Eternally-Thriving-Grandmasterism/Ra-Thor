/// Rule: Check for presence of critical IDs
///
/// Truth: Certain elements (like language switchers, main containers)
/// must have stable IDs for JavaScript and accessibility to work reliably.

pub fn check(html: &str) -> Vec<String> {
    let mut issues = vec![];

    if !html.contains("id=\"lang-selector\"") {
        issues.push("Missing required id='lang-selector' for language switching".to_string());
    }

    issues
}
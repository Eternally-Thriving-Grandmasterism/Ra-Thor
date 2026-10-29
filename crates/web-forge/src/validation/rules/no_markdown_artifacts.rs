/// Rule: Detect leftover markdown artifacts (e.g. **Summary:**)
/// This was one of the most common issues during early development.

pub fn check(html: &str) -> Vec<String> {
    let mut issues = vec![];

    if html.contains("**Summary:**") {
        issues.push("Detected leftover markdown '**Summary:**' in HTML".to_string());
    }

    issues
}
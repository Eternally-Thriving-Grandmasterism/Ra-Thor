/// Rule: Ensure <details> elements have proper <summary> children

pub fn check(html: &str) -> Vec<String> {
    let mut issues = vec![];

    // Simple heuristic check
    if html.contains("<details") && !html.contains("<summary>") {
        issues.push("Found <details> without a <summary> child".to_string());
    }

    issues
}
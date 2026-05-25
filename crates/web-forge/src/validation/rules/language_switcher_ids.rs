/// Rule: Check for presence of language switcher IDs
/// Helps ensure the multi-language system works correctly.

pub fn check(html: &str) -> Vec<String> {
    let mut issues = vec![];

    if !html.contains("lang-selector") {
        issues.push("Missing language switcher container (id='lang-selector')".to_string());
    }

    issues
}
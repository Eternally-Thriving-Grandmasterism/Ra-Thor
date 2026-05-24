/// Basic Accessibility Rule
///
/// Truth: Even basic accessibility checks dramatically improve
/// the quality and inclusivity of generated websites.

pub fn check(html: &str) -> Vec<String> {
    let mut issues = vec![];

    if !html.contains("alt=\"") && html.contains("<img") {
        issues.push("Images found without alt attributes (basic accessibility)".to_string());
    }

    if !html.contains("<label") && html.contains("<input") {
        issues.push("Form inputs may be missing associated labels".to_string());
    }

    issues
}
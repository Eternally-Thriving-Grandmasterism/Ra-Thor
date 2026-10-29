/// Example: Using the HtmlValidator Engine
///
/// This demonstrates how to validate HTML using web-forge's
/// Validation Engine with multiple rules.

use web_forge::validation::HtmlValidator;

fn main() {
    let html = r#"
        <div>
            <details>
                <summary>Click me</summary>
                Content here
            </details>
        </div>
    "#;

    let validator = HtmlValidator::new();
    let issues = validator.validate(html);

    if issues.is_empty() {
        println!("✅ HTML is valid!");
    } else {
        println!("❌ Validation issues found:");
        for issue in issues {
            println!(" - {}", issue);
        }
    }
}
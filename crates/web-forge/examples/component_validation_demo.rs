use web_forge::validation::HtmlValidator;

fn main() {
    let mut validator = HtmlValidator::new();

    println!("Core components registered with validator.");

    let sample_html = r#"<button class=\"btn btn-Primary\">Click</button>"#;
    let issues = validator.validate(sample_html);

    if issues.is_empty() {
        println!("✅ HTML passed validation");
    } else {
        println!("❌ Issues found:");
        for issue in issues {
            println!(" - {}", issue);
        }
    }
}
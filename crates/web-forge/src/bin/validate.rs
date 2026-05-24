/// Simple CLI tool for the web-forge Validation Engine
///
/// Usage: cargo run --bin validate -- "<html string>"

use web_forge::validation::HtmlValidator;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: validate <html>");
        std::process::exit(1);
    }

    let html = &args[1];
    let validator = HtmlValidator::new();
    let issues = validator.validate(html);

    if issues.is_empty() {
        println!("✅ Valid HTML");
    } else {
        println!("❌ Issues found:");
        for issue in issues {
            println!("- {}", issue);
        }
    }
}
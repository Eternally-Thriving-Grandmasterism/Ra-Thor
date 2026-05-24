/// Full End-to-End Phase 2 Orchestration Demo
///
/// Demonstrates the complete pipeline:
/// Prompt → ComponentAwareGenerator → ComponentTree → Renderer → HTML + Validation

use web_forge::orchestration::generation::ComponentAwareGenerator;
use web_forge::orchestration::validation_loop::RefiningValidationLoop;
use web_forge::orchestration::renderer::render_tree;
use web_forge::orchestration::ValidationFeedbackLoop;
use web_forge::validation::HtmlValidator;
use web_forge::orchestration::component_tree::ComponentTree;
use serde_json::from_str;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Phase 2: Full Component-Aware Orchestration Demo       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let generator = ComponentAwareGenerator::new();
    let validation_loop = RefiningValidationLoop::new(generator)
        .with_max_attempts(3);

    let validator = HtmlValidator::new();

    let prompt = "Create a primary call-to-action button";
    println!("Prompt: {}\n", prompt);

    // Run the full generate → validate → refine loop
    let result = validation_loop.run(prompt, &validator);

    println!("─── ComponentTree (JSON) ───");
    println!("{}\n", result.generated_html);

    // Try to parse and render the ComponentTree
    if let Ok(tree) = from_str::<ComponentTree>(&result.generated_html) {
        let html_output = render_tree(&tree);

        println!("─── Rendered HTML ───");
        println!("{}\n", html_output);
    } else {
        println!("[Warning] Could not parse output as ComponentTree for rendering.\n");
    }

    println!("─── Validation Result ───");
    println!("Success: {}", result.success);
    if !result.validation_issues.is_empty() {
        println!("Issues:");
        for issue in &result.validation_issues {
            println!("  - {}", issue);
        }
    }

    println!("\n=== Demo Complete ===");
}

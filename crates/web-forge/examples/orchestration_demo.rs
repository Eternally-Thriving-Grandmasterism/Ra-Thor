/// Orchestration Demo
///
/// Demonstrates integration between ComponentAwareGenerator and RefiningValidationLoop.

use web_forge::orchestration::generation::ComponentAwareGenerator;
use web_forge::orchestration::validation_loop::RefiningValidationLoop;
use web_forge::orchestration::ValidationFeedbackLoop;
use web_forge::validation::HtmlValidator;

fn main() {
    println!("=== Phase 2 Orchestration Demo ===\n");

    let generator = ComponentAwareGenerator::new();
    let validation_loop = RefiningValidationLoop::new(generator)
        .with_max_attempts(3);

    let validator = HtmlValidator::new();

    let prompt = "Create a primary action button";
    println!("Prompt: {}\n", prompt);

    let result = validation_loop.run(prompt, &validator);

    println!("Generated Output (ComponentTree JSON):\n{}", result.generated_html);
    println!("\nValidation Issues: {:?}", result.validation_issues);
    println!("Success: {}", result.success);
}

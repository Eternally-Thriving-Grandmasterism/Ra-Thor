/// Example: Registering components with the Validation Engine
///
/// This shows how components can be registered so the validator
/// knows which components are valid.

use web_forge::validation::HtmlValidator;

fn main() {
    let mut validator = HtmlValidator::new();

    // Register current components
    validator.register_component("Button");
    validator.register_component("Card");
    validator.register_component("Input");
    validator.register_component("Modal");

    println!("Components registered with validator.");
}
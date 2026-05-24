/// Validation Feedback Loop Module
///
/// Implements the core generate → validate → refine cycle.
///
/// This is one of the most important parts of high-quality AI orchestration.

use crate::orchestration::{GenerationStrategy, ValidationFeedbackLoop, OrchestrationResult};
use crate::validation::HtmlValidator;

/// A basic implementation of a validation feedback loop.
pub struct BasicValidationLoop<G: GenerationStrategy> {
    generator: G,
}

impl<G: GenerationStrategy> BasicValidationLoop<G> {
    pub fn new(generator: G) -> Self {
        Self { generator }
    }
}

impl<G: GenerationStrategy> ValidationFeedbackLoop for BasicValidationLoop<G> {
    fn run(&self, prompt: &str, validator: &HtmlValidator) -> OrchestrationResult {
        // Step 1: Generate
        let generated = self.generator.generate(prompt);

        // Step 2: Validate (sanitization happens inside validator)
        let issues = validator.validate(&generated);

        let success = issues.is_empty();

        OrchestrationResult {
            generated_html: generated,
            validation_issues: issues,
            success,
        }
    }
}

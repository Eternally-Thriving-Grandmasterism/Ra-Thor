/// Validation Feedback Loop Module
///
/// Implements generate → validate → refine with retry logic.

use crate::orchestration::{GenerationStrategy, ValidationFeedbackLoop, OrchestrationResult};
use crate::validation::HtmlValidator;

/// A robust validation loop with retry/refinement support.
pub struct RefiningValidationLoop<G: GenerationStrategy> {
    generator: G,
    max_attempts: usize,
}

impl<G: GenerationStrategy> RefiningValidationLoop<G> {
    pub fn new(generator: G) -> Self {
        Self {
            generator,
            max_attempts: 3,
        }
    }

    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }
}

impl<G: GenerationStrategy> ValidationFeedbackLoop for RefiningValidationLoop<G> {
    fn run(&self, prompt: &str, validator: &HtmlValidator) -> OrchestrationResult {
        let mut current_prompt = prompt.to_string();

        for attempt in 1..=self.max_attempts {
            let generated = self.generator.generate(&current_prompt);
            let issues = validator.validate(&generated);
            let success = issues.is_empty();

            if success {
                return OrchestrationResult {
                    generated_html: generated,
                    validation_issues: issues,
                    success: true,
                };
            }

            if attempt == self.max_attempts {
                // Final attempt failed
                return OrchestrationResult {
                    generated_html: generated,
                    validation_issues: issues,
                    success: false,
                };
            }

            // Prepare refined prompt with feedback
            current_prompt = format!(
                "Original request: {}.\nPrevious attempt had these issues: {}.\nPlease generate an improved version.",
                prompt,
                issues.join("; ")
            );
        }

        // Should not reach here
        OrchestrationResult {
            generated_html: String::new(),
            validation_issues: vec!["Unexpected failure in refinement loop".to_string()],
            success: false,
        }
    }
}

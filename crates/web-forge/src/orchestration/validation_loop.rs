/// Validation Feedback Loop Module
///
/// Implements generate → validate → refine with retry logic and logging.

use crate::orchestration::{GenerationStrategy, ValidationFeedbackLoop, OrchestrationResult};
use crate::validation::HtmlValidator;

/// A robust validation loop with retry/refinement + logging.
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
        println!("[Orchestration] Starting refinement loop (max_attempts={})...", self.max_attempts);

        let mut current_prompt = prompt.to_string();

        for attempt in 1..=self.max_attempts {
            println!("[Orchestration] Attempt {}/{}...", attempt, self.max_attempts);

            let generated = self.generator.generate(&current_prompt);
            let issues = validator.validate(&generated);
            let success = issues.is_empty();

            if success {
                println!("[Orchestration] Success on attempt {}", attempt);
                return OrchestrationResult {
                    generated_html: generated,
                    validation_issues: issues,
                    success: true,
                };
            }

            if attempt == self.max_attempts {
                println!("[Orchestration] Failed after {} attempts", self.max_attempts);
                return OrchestrationResult {
                    generated_html: generated,
                    validation_issues: issues,
                    success: false,
                };
            }

            // Build better feedback prompt
            let feedback = issues.join("; ");
            println!("[Orchestration] Issues found: {}", feedback);

            current_prompt = format!(
                "Original request: {}.\nPrevious issues: {}.\nPlease fix the problems and generate an improved version.",
                prompt,
                feedback
            );
        }

        OrchestrationResult {
            generated_html: String::new(),
            validation_issues: vec!["Unexpected failure".to_string()],
            success: false,
        }
    }
}

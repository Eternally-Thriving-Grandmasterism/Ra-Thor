/// Validation Feedback Loop Module
///
/// Implements the core generate → validate → refine cycle.
/// This is one of the most important parts of high-quality AI orchestration.

use crate::orchestration::{GenerationStrategy, ValidationFeedbackLoop, OrchestrationResult};
use crate::validation::HtmlValidator;

/// A validation loop with basic refinement support.
pub struct RefiningValidationLoop<G: GenerationStrategy> {
    generator: G,
    max_refinements: usize,
}

impl<G: GenerationStrategy> RefiningValidationLoop<G> {
    pub fn new(generator: G) -> Self {
        Self {
            generator,
            max_refinements: 2,
        }
    }

    pub fn with_max_refinements(mut self, max: usize) -> Self {
        self.max_refinements = max;
        self
    }
}

impl<G: GenerationStrategy> ValidationFeedbackLoop for RefiningValidationLoop<G> {
    fn run(&self, prompt: &str, validator: &HtmlValidator) -> OrchestrationResult {
        let mut current_prompt = prompt.to_string();

        for attempt in 0..=self.max_refinements {
            let generated = self.generator.generate(&current_prompt);
            let issues = validator.validate(&generated);
            let success = issues.is_empty();

            if success || attempt == self.max_refinements {
                return OrchestrationResult {
                    generated_html: generated,
                    validation_issues: issues,
                    success,
                };
            }

            // Simple feedback for next attempt
            current_prompt = format!(
                "{}. Please fix these issues: {}",
                prompt,
                issues.join("; ")
            );
        }

        OrchestrationResult {
            generated_html: String::new(),
            validation_issues: vec!["Max refinements reached".to_string()],
            success: false,
        }
    }
}

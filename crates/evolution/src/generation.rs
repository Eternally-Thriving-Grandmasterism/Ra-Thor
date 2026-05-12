/// TOLC + Mercy-aware proposal generation module (Phase B).
///
/// Strengthens the generation side of the self-improvement loop.
/// All generation is guided by TOLC principles and Mercy Gate considerations.

use llama_cpp_gguf::{generate_chat, ChatMessage, GenerationConfig, LlamaModel};
use tracing::info;

/// Generate a proposal using LLM with explicit TOLC + Mercy guidance.
pub fn generate_proposal(
    model: &LlamaModel,
    topic: &str,
    context: Option<&str>,
) -> String {
    info!(
        topic = topic,
        has_context = context.is_some(),
        "Generating proposal with TOLC/Mercy guidance"
    );

    let context_str = context.unwrap_or("No additional context provided.");

    let prompt = format!(
        r#"You are a proposal generator for Rathor.ai's self-evolution system.

Generate a clear, well-structured proposal on the following topic.

**Topic:** {}

**Context:** {}

**Requirements:**
- Be truthful, orderly, logical, and compassionate (TOLC principles)
- Respect sovereignty and avoid harm
- Promote harmony and positive impact
- Keep the proposal focused and actionable

Respond with only the proposal text, no extra commentary."#,
        topic, context_str
    );

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: prompt,
    }];

    generate_chat(model, &messages, &GenerationConfig::default())
        .unwrap_or_else(|_| format!("Failed to generate proposal on: {}", topic))
}

/// Generate a proposal and immediately evaluate it using the full TOLC + Mercy pipeline.
/// Returns both the generated proposal and its evaluation result.
pub fn generate_and_evaluate(
    model: &LlamaModel,
    topic: &str,
    context: Option<&str>,
) -> (String, crate::evaluation::EvaluationResult) {
    let proposal = generate_proposal(model, topic, context);
    let evaluation = crate::evaluation::evaluate_proposal(model, &proposal);

    info!(
        topic = topic,
        is_acceptable = evaluation.is_acceptable(),
        average_tolc = evaluation.average_tolc_score,
        average_mercy = evaluation.average_mercy_score,
        "Generated and evaluated proposal"
    );

    (proposal, evaluation)
}

/// Generate multiple proposal variations on the same topic.
pub fn generate_proposal_variations(
    model: &LlamaModel,
    topic: &str,
    count: usize,
) -> Vec<String> {
    info!(topic = topic, count = count, "Generating proposal variations");

    (0..count)
        .map(|i| {
            let variation_context = format!("Variation #{}", i + 1);
            generate_proposal(model, topic, Some(&variation_context))
        })
        .collect()
}

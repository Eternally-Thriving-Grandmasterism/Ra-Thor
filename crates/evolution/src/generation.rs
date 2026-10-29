/// Enhanced TOLC + Mercy-aware proposal generation (Phase B + Research Integration).
///
/// Incorporates techniques from Darwin Gödel Machine, AlphaEvolve, reflective generation,
/// and RAG grounding for higher-quality, context-aware proposals.

use llama_cpp_gguf::{generate_chat, ChatMessage, GenerationConfig, LlamaModel};
use tracing::{info, debug};

/// Structured proposal output schema (enforces quality and safety).
#[derive(Debug, Clone)]
pub struct Proposal {
    pub title: String,
    pub rationale: String,
    pub changes: String,           // Could be diff, plan, or description
    pub expected_impact: String,
    pub mercy_alignment: String,   // How it aligns with Mercy Gates
    pub tolc_alignment: String,    // How it aligns with TOLC
}

/// Enhanced proposal generation with reflection + RAG-style grounding.
/// Uses `plan.md` content + topic as context for more relevant proposals.
pub fn generate_proposal(
    model: &LlamaModel,
    topic: &str,
    context: Option<&str>,
) -> String {
    info!(
        topic = topic,
        has_context = context.is_some(),
        "Generating enhanced TOLC/Mercy proposal with reflection + RAG grounding"
    );

    let context_str = context.unwrap_or("No additional context provided.");

    // Reflective prompt inspired by Darwin Gödel Machine / reflective agents
    let prompt = format!(
        r#"You are an advanced proposal generator for Rathor.ai's self-evolving lattice.

**Task:** Generate a high-quality, actionable proposal on the given topic.

**Context (RAG-grounded from plan.md + system state):**
{}

**Topic:** {}

**Instructions (reflect before proposing):**
1. First, briefly reflect on what would most improve the system in this area.
2. Then generate a structured proposal that is:
   - Truthful, Orderly, Logical, and Compassionate (TOLC)
   - Respectful of Sovereignty, Non-Harm, and Harmony (Mercy Gates)
   - Specific and implementable

**Output ONLY valid JSON** in this exact format (no extra text):
{{
  "title": "Short descriptive title",
  "rationale": "Why this proposal matters and what problem it solves",
  "changes": "Detailed description of what should change (or code diff if applicable)",
  "expected_impact": "Expected positive outcomes and metrics",
  "mercy_alignment": "How this respects the 7 Living Mercy Gates",
  "tolc_alignment": "How this aligns with TOLC principles"
}}"#,
        context_str, topic
    );

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: prompt,
    }];

    let response = generate_chat(model, &messages, &GenerationConfig::default())
        .unwrap_or_else(|_| format!("Failed to generate proposal on: {}", topic));

    debug!(response_len = response.len(), "Raw LLM response received");
    response
}

/// Generate a proposal and immediately evaluate it.
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
        "Generated and evaluated proposal (enhanced)"
    );

    (proposal, evaluation)
}

/// Generate multiple diverse proposal variations.
pub fn generate_proposal_variations(
    model: &LlamaModel,
    topic: &str,
    count: usize,
) -> Vec<String> {
    info!(topic = topic, count = count, "Generating diverse proposal variations");

    (0..count)
        .map(|i| {
            let variation_context = format!("Variation #{} - explore a different angle", i + 1);
            generate_proposal(model, topic, Some(&variation_context))
        })
        .collect()
}

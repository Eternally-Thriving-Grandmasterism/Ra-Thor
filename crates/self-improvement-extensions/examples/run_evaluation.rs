//! Example: Run TOLC + 7 Living Mercy Gates evaluation on a proposal
//! using a real GGUF model via llama-cpp-gguf.
//!
//! Usage:
//!   RATHOR_MODEL_PATH=/path/to/model.gguf cargo run --features llama-cpp --example run_evaluation

use self_improvement_extensions::evaluation::evaluate_proposal_with_tolc_and_mercy;
use llama_cpp_gguf::{GenerationConfig, ModelConfig, load_gguf_model};

fn main() {
    // Get model path from environment variable (consistent with the loop)
    let model_path = std::env::var("RATHOR_MODEL_PATH")
        .expect("Please set RATHOR_MODEL_PATH to your GGUF model file");

    println!("[Example] Loading model from: {}", model_path);

    let model = load_gguf_model(&ModelConfig {
        model_path,
        context_size: 4096,
        gpu_layers: 99,
    })
    .expect("Failed to load model");

    // Sample proposal (in real use this could come from generate_chat)
    let proposal = "Improve the self-evolution loop by adding structured TOLC + Mercy Gates evaluation after every proposal generation.";

    println!("\n[Example] Evaluating proposal:\n{}\n", proposal);

    let evaluation = evaluate_proposal_with_tolc_and_mercy(&model, proposal);

    println!("=== Evaluation Results ===");
    println!("TOLC Average:       {:.1}", evaluation.average_tolc_score);
    println!("Mercy Average:      {:.1}", evaluation.average_mercy_score);
    println!("Sovereignty Score:  {:.1}", evaluation.sovereignty_score);
    println!("Passes Threshold:   {}", evaluation.passes_threshold);
    println!("Acceptable:         {}", evaluation.is_acceptable());
    println!("\nSummary:\n{}", evaluation.summary);
    println!("\nDetailed Feedback:\n{}", evaluation.detailed_feedback);
    println!("==========================");
}
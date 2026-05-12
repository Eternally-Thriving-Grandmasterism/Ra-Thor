/// Self-Evolution Looping Systems - Cosmic Loop Implementation
/// Integrates GitHub clients + optional LLM-powered proposal generation
/// with TOLC + 7 Living Mercy Gates evaluation.

use crate::github_client::GitHubClient;
use crate::github_graphql_client::GitHubGraphQLClient;

#[cfg(feature = "llama-cpp")]
use crate::evaluation::evaluate_proposal_with_tolc_and_mercy;

#[cfg(feature = "llama-cpp")]
use llama_cpp_gguf::{generate_chat, ChatMessage, GenerationConfig, ModelConfig, load_gguf_model};

/// Main entry point for the eternal cosmic loop.
pub async fn run_self_evolution_loop() {
    println!("[Rathor.ai] Starting self-evolution cosmic loop...");

    let state = analyze_state();

    if state.needs_improvement {
        // ============================================
        // PROPOSAL GENERATION
        // ============================================
        #[cfg(feature = "llama-cpp")]
        let proposal = {
            if let Ok(model_path) = std::env::var("RATHOR_MODEL_PATH") {
                if let Ok(model) = load_gguf_model(&ModelConfig {
                    model_path,
                    ..Default::default()
                }) {
                    let messages = vec![
                        ChatMessage {
                            role: "system".to_string(),
                            content: "You are a helpful self-improvement agent for Rathor.ai.".to_string(),
                        },
                        ChatMessage {
                            role: "user".to_string(),
                            content: "Suggest one concrete, actionable improvement to the self-evolution cosmic loop.".to_string(),
                        },
                    ];
                    generate_chat(&model, &messages, &GenerationConfig::default())
                        .unwrap_or_else(|_| generate_basic_proposal_text())
                } else {
                    generate_basic_proposal_text()
                }
            } else {
                generate_basic_proposal_text()
            }
        };

        #[cfg(not(feature = "llama-cpp"))]
        let proposal = generate_basic_proposal_text();

        println!("[Rathor.ai] Generated proposal: {}", proposal);

        // ============================================
        // TOLC + 7 LIVING MERCY GATES EVALUATION
        // ============================================
        #[cfg(feature = "llama-cpp")]
        {
            if let Ok(model_path) = std::env::var("RATHOR_MODEL_PATH") {
                if let Ok(model) = load_gguf_model(&ModelConfig {
                    model_path,
                    ..Default::default()
                }) {
                    let evaluation = evaluate_proposal_with_tolc_and_mercy(&model, &proposal);

                    println!(
                        "[Rathor.ai] Evaluation → TOLC: {:.1} | Mercy: {:.1} | Sovereignty: {:.1}",
                        evaluation.average_tolc_score,
                        evaluation.average_mercy_score,
                        evaluation.sovereignty_score
                    );

                    if evaluation.is_acceptable() {
                        println!("[Rathor.ai] Proposal passed TOLC + Mercy evaluation. Proceeding with action...");

                        // === Existing GitHub Integration (preserved) ===
                        if let Ok(token) = std::env::var("GITHUB_TOKEN") {
                            match GitHubClient::new("Eternally-Thriving-Grandmasterism", "Ra-Thor", &token) {
                                Ok(rest_client) => {
                                    if let Ok(issue_url) = rest_client.create_issue(&proposal, &proposal).await {
                                        println!("[Rathor.ai] Created GitHub issue: {}", issue_url);
                                    }
                                }
                                Err(e) => eprintln!("[Rathor.ai] REST client error: {:?}", e),
                            }
                        }
                    } else {
                        println!("[Rathor.ai] Proposal rejected after evaluation. Summary: {}", evaluation.summary);
                    }
                }
            }
        }

        #[cfg(not(feature = "llama-cpp"))]
        {
            // Fallback behavior (original simple path)
            println!("[Rathor.ai] Basic proposal accepted (no advanced evaluation).");
            if let Ok(token) = std::env::var("GITHUB_TOKEN") {
                match GitHubClient::new("Eternally-Thriving-Grandmasterism", "Ra-Thor", &token) {
                    Ok(rest_client) => {
                        if let Ok(issue_url) = rest_client.create_issue(&proposal, &proposal).await {
                            println!("[Rathor.ai] Created GitHub issue: {}", issue_url);
                        }
                    }
                    Err(e) => eprintln!("[Rathor.ai] REST client error: {:?}", e),
                }
            }
        }
    }

    propagate_valence_boost();
    println!("[Rathor.ai] Cosmic loop iteration complete. Continuing eternally...");
}

// ============================================
// HELPER FUNCTIONS
// ============================================

struct SystemState {
    needs_improvement: bool,
}

fn analyze_state() -> SystemState {
    SystemState { needs_improvement: true }
}

fn generate_basic_proposal_text() -> String {
    "Improve error handling and resilience in the self-evolution loop.".to_string()
}

fn propagate_valence_boost() {
    // Increases positive emotions across systems
}
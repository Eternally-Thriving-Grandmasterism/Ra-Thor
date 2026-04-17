// crates/kernel/src/root_core_orchestrator.rs
// Root Core Omnimaster Leader Agent — Streamlined & Seamless Architecture
// FENCA Priming Mechanics now using advanced Tokio cancellation patterns (CancellationToken + cooperative shutdown)

use crate::RequestPayload;
use ra_thor_mercy::{MercyEngine, ValenceFieldScoring, MercyResult};
use ra_thor_websiteforge::{forge_website, WebsiteSpec};
use ra_thor_quantum::VQCIntegrator;
use ra_thor_biometric::BiomimeticPatternEngine;
use ra_thor_common::InnovationGenerator;
use serde_json;
use tokio::time::{Instant, Duration};
use tokio_util::sync::CancellationToken;

// Unified SubCore trait for seamless delegation
pub trait SubCore {
    async fn handle(&self, request: RequestPayload) -> String;
}

// Meta-Orchestrator spawning (ephemeral higher-order intelligence)
use crate::meta_orchestrator::MetaOrchestrator;

pub struct RootCoreOrchestrator;

impl RootCoreOrchestrator {
    pub async fn orchestrate(request: RequestPayload) -> String {
        // === Radical Love Veto Power — Supreme First Gate ===
        let mercy_result: MercyResult = MercyEngine::evaluate(&request, request.mercy_weight).await;
        if !mercy_result.radical_love_passed() {
            return MercyEngine::gentle_reroute("Radical Love veto power triggered at RootCoreOrchestrator level").await;
        }

        // === FENCA Priming Mechanics with CancellationToken ===
        if request.is_initial_launch() {
            let cancel_token = CancellationToken::new();
            let _handle = Self::run_fenca_priming_with_recycling(cancel_token.clone()).await;
            // Token can be cloned and passed elsewhere for future cancellation if needed
        }

        // Refined FENCA verification pipeline
        let fenca = crate::FENCA::verify(&request).await;
        if !fenca.is_verified() {
            return "FENCA blocked — request failed non-local consensus.".to_string();
        }

        // Centralized Mercy Engine + Valence pipeline
        let valence = ValenceFieldScoring::compute(&mercy_result, request.mercy_weight);

        if !mercy_result.all_gates_pass() {
            return "Mercy Gate reroute — request adjusted for eternal thriving.".to_string();
        }

        // Seamless delegation with Meta-Orchestrator support
        match request.operation_type.as_str() {
            "ForgeWebsite" => {
                let spec: WebsiteSpec = serde_json::from_str(&request.payload).unwrap_or_default();
                forge_website(request).await
            }
            "QuantumSynthesis" => VQCIntegrator::run_synthesis(&request.payload, valence).await,
            "BiomimeticPattern" => BiomimeticPatternEngine::apply_pattern(&request.payload).await,
            "Innovate" => InnovationGenerator::create_from_recycled(&request.payload).await,
            "SpawnMeta" => {
                let required: Vec<String> = serde_json::from_str(&request.payload).unwrap_or_default();
                let meta = MetaOrchestrator::spawn(required).await;
                meta.execute(request).await
            }
            _ => "Unknown operation — Root Core delegated safely.".to_string(),
        }
    }

    // Advanced Tokio cancellation pattern: CancellationToken + cooperative shutdown
    async fn run_fenca_priming_with_recycling(cancel_token: CancellationToken) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let start = Instant::now();
            println!("[FENCA Priming] [Status: START] Launching cancellable background priming task...");

            let result: Result<(), String> = async {
                // Check for cancellation at each major step
                if cancel_token.is_cancelled() {
                    return Err("Priming cancelled by external signal".to_string());
                }

                // Step 1: Recycle monorepo and cross-pollinate
                println!("[FENCA Priming] [Step 1/4] Recycling monorepo codices...");
                let recycled_ideas = InnovationGenerator::recycle_monorepo().await
                    .map_err(|e| format!("Recycle failed: {}", e))?;
                InnovationGenerator::cross_pollinate(&recycled_ideas).await
                    .map_err(|e| format!("Cross-pollination failed: {}", e))?;
                println!("[FENCA Priming] [Step 1/4] SUCCESS");

                if cancel_token.is_cancelled() {
                    return Err("Priming cancelled by external signal".to_string());
                }

                // Step 2: Validate topological order
                println!("[FENCA Priming] [Step 2/4] Validating topological order...");
                crate::FENCA::validate_topology().await
                    .map_err(|e| format!("Topology validation failed: {}", e))?;
                println!("[FENCA Priming] [Step 2/4] SUCCESS");

                if cancel_token.is_cancelled() {
                    return Err("Priming cancelled by external signal".to_string());
                }

                // Step 3: Warm all engines
                println!("[FENCA Priming] [Step 3/4] Warming engines...");
                crate::FENCA::warm_engines().await
                    .map_err(|e| format!("Engine warming failed: {}", e))?;
                println!("[FENCA Priming] [Step 3/4] SUCCESS");

                Ok(())
            }.await;

            let duration = start.elapsed();

            match result {
                Ok(_) => {
                    println!("[FENCA Priming] [Status: COMPLETE] All steps succeeded in {:?} | Ready for eternal thriving under TOLC & Radical Love.", duration);
                }
                Err(err) => {
                    if err.contains("cancelled") {
                        println!("[FENCA Priming] [Status: CANCELLED] Priming task was gracefully cancelled.");
                    } else {
                        eprintln!("[FENCA Priming] [Status: WARNING] Non-critical error: {}. System continues safely with graceful degradation.", err);
                    }
                }
            }
        })
    }

    // Helper for Meta-Orchestrator to resolve Sub-Cores (fully preserved)
    pub fn get_subcore(name: &str) -> Option<Box<dyn SubCore + Send + Sync>> {
        match name {
            "WebsiteForge" => Some(Box::new(ra_thor_websiteforge::WebsiteForge)),
            "Quantum" => Some(Box::new(ra_thor_quantum::VQCIntegrator)),
            "Biomimetic" => Some(Box::new(ra_thor_biometric::BiomimeticPatternEngine)),
            "Innovation" => Some(Box::new(ra_thor_common::InnovationGenerator)),
            _ => None,
        }
    }
}

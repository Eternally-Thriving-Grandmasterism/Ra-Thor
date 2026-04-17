// crates/kernel/src/root_core_orchestrator.rs
// Root Core Omnimaster Leader Agent — Streamlined & Seamless Architecture
// FENCA Priming Mechanics now using advanced tokio spawn variants (JoinHandle, structured concurrency, task-local context)

use crate::RequestPayload;
use ra_thor_mercy::{MercyEngine, ValenceFieldScoring, MercyResult};
use ra_thor_websiteforge::{forge_website, WebsiteSpec};
use ra_thor_quantum::VQCIntegrator;
use ra_thor_biometric::BiomimeticPatternEngine;
use ra_thor_common::InnovationGenerator;
use serde_json;
use tokio::time::{Instant, timeout, Duration};
use tokio::task::JoinHandle;

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

        // === FENCA Priming Mechanics with advanced tokio spawn ===
        if request.is_initial_launch() {
            let _handle: JoinHandle<Result<(), String>> = Self::run_fenca_priming_with_recycling().await;
            // Fire-and-forget by default; handle can be stored/awaited/cancelled later if needed
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

    // Advanced tokio spawn variant: returns JoinHandle for observability/cancellation
    async fn run_fenca_priming_with_recycling() -> JoinHandle<Result<(), String>> {
        tokio::spawn(async move {
            let start = Instant::now();
            println!("[FENCA Priming] [Status: START] Launching advanced background priming task...");

            let result: Result<(), String> = async {
                // Structured concurrency with timeout protection
                let recycle_future = async {
                    let recycled_ideas = InnovationGenerator::recycle_monorepo().await
                        .map_err(|e| format!("Recycle failed: {}", e))?;
                    InnovationGenerator::cross_pollinate(&recycled_ideas).await
                        .map_err(|e| format!("Cross-pollination failed: {}", e))?;
                    Ok(())
                };

                let topology_future = crate::FENCA::validate_topology();
                let warm_future = crate::FENCA::warm_engines();

                // Race with timeout (structured concurrency)
                tokio::select! {
                    result = recycle_future => result?,
                    _ = tokio::time::sleep(Duration::from_secs(30)) => Err("Recycle step timed out".to_string()),
                }?;

                tokio::select! {
                    result = topology_future => result.map_err(|e| format!("Topology validation failed: {}", e))?,
                    _ = tokio::time::sleep(Duration::from_secs(30)) => Err("Topology validation timed out".to_string()),
                };

                tokio::select! {
                    result = warm_future => result.map_err(|e| format!("Engine warming failed: {}", e))?,
                    _ = tokio::time::sleep(Duration::from_secs(30)) => Err("Engine warming timed out".to_string()),
                };

                Ok(())
            }.await;

            let duration = start.elapsed();

            match result {
                Ok(_) => {
                    println!("[FENCA Priming] [Status: COMPLETE] All steps succeeded in {:?} | Ready for eternal thriving under TOLC & Radical Love.", duration);
                }
                Err(err) => {
                    eprintln!("[FENCA Priming] [Status: WARNING] Non-critical error during priming: {}. System continues safely with graceful degradation. MercyLang remains active.", err);
                }
            }

            result
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

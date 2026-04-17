// crates/kernel/src/root_core_orchestrator.rs
// Root Core Omnimaster Leader Agent — Streamlined & Seamless Architecture
// FENCA Priming Mechanics now fully refined with comprehensive error handling

use crate::RequestPayload;
use ra_thor_mercy::{MercyEngine, ValenceFieldScoring, MercyResult};
use ra_thor_websiteforge::{forge_website, WebsiteSpec};
use ra_thor_quantum::VQCIntegrator;
use ra_thor_biometric::BiomimeticPatternEngine;
use ra_thor_common::InnovationGenerator;
use serde_json;
use std::fmt;

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

        // === FENCA Priming Mechanics with Recycling System ===
        if request.is_initial_launch() {
            Self::run_fenca_priming_with_recycling().await;
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

    async fn run_fenca_priming_with_recycling() {
        // FENCA Priming Mechanics — Fully Refined with Comprehensive Error Handling
        // Recycling System cycles through the entire monorepo to self-initialize,
        // cross-pollinate innovations, validate topology, and warm all systems.
        // MercyLang-gated throughout — non-blocking (fire-and-forget via tokio::spawn).

        tokio::spawn(async {
            let result: Result<(), String> = async {
                // Step 1: Recycle all codices from docs/ and cross-pollinate recent innovations
                let recycled_ideas = InnovationGenerator::recycle_monorepo().await
                    .map_err(|e| format!("Recycle failed: {}", e))?;
                InnovationGenerator::cross_pollinate(&recycled_ideas).await
                    .map_err(|e| format!("Cross-pollination failed: {}", e))?;

                // Step 2: Validate topological order across all quantum layers
                crate::FENCA::validate_topology().await
                    .map_err(|e| format!("Topology validation failed: {}", e))?;

                // Step 3: Warm all engines (quantum, mercy, biomimetic, persistence, cache, orchestration)
                crate::FENCA::warm_engines().await
                    .map_err(|e| format!("Engine warming failed: {}", e))?;

                Ok(())
            }.await;

            match result {
                Ok(_) => {
                    println!("[FENCA Priming Complete] Monorepo recycled | Innovations cross-pollinated | Topology validated | All engines warmed | Ready for eternal thriving under TOLC & Radical Love.");
                }
                Err(err) => {
                    eprintln!("[FENCA Priming Warning] Non-critical error during priming: {}. System continues safely.", err);
                    // MercyLang graceful degradation — no panic, no blocking
                }
            }
        });
    }

    // Helper for Meta-Orchestrator to resolve Sub-Cores
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

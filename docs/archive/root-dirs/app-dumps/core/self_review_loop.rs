// core/self_review_loop.rs
// Self-Review Loop — The Omnimaster Root Core's living self-improvement engine
// Scans /docs, recycles ideas, generates innovations, and delegates to Sub-Cores
// Everything is FENCA-verified, mercy-gated, auditable, and mercy-weighted

use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use crate::global_cache::GlobalCache;
use crate::root_core_orchestrator::RootCoreOrchestrator;
use crate::codex_loader::CodexLoader;
use crate::idea_recycler::IdeaRecycler;
use crate::innovation_generator::InnovationGenerator;
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct SelfReviewLoop;

impl SelfReviewLoop {
    /// Main self-review loop — the Omnimaster's living self-improvement heartbeat
    pub async fn run() {
        let start = SystemTime::now();

        // 1. Scan /docs folder for new or updated codices (returns filenames)
        let codex_filenames = CodexLoader::scan_docs_folder().await;

        let mut processed = 0usize;
        let mut total_recycled = 0usize;

        for filename in &codex_filenames {
            // 1b. Load actual content (FENCA + Mercy already applied inside loader)
            let Some(content) = CodexLoader::load_codex(filename).await else {
                continue;
            };

            // 2. FENCA — primordial truth gate (re-confirm on loaded content)
            let fenca_result = FENCA::verify_codex_content(&content).await;
            if !fenca_result.is_verified() {
                continue;
            }

            // 3. Mercy Engine evaluation
            let mercy_scores = MercyEngine::evaluate_codex_content(&content);
            let valence = ValenceFieldScoring::calculate(&mercy_scores);
            if !mercy_scores.all_gates_pass() {
                continue;
            }

            // 4. Mercy Weight for this review
            let mercy_weight = MercyWeighting::derive_mercy_weight(
                valence,
                fenca_result.fidelity(),
                None,
                /* context */
            );

            // 5. Recycle useful ideas → structured RecycledIdea, then convert to seeds
            //    for the current InnovationGenerator API (clean bidirectional handoff)
            let recycled_seeds =
                IdeaRecycler::extract_and_recycle_as_seeds(&content, mercy_weight).await;
            total_recycled += recycled_seeds.len();

            if recycled_seeds.is_empty() {
                continue;
            }

            // 6. Generate innovation from recycled ideas
            if let Some(innovation) = InnovationGenerator::create_from_recycled(
                recycled_seeds,
                &mercy_scores,
                mercy_weight,
            )
            .await
            {
                // 7. Delegate innovation to the appropriate Sub-Core
                RootCoreOrchestrator::delegate_innovation(innovation).await;
                processed += 1;
            }
        }

        // 8. Self-optimization of mercy weighting and parameters
        MercyWeighting::tune_from_recent_audit_logs().await;

        // 9. Immutable audit of the entire self-review loop
        let duration_ms = start
            .elapsed()
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let _ = AuditLogger::log(
            "root",
            None,
            "self_review_loop",
            "complete",
            true,
            1.0,
            1.0,
            vec![],
            json!({
                "codices_scanned": codex_filenames.len(),
                "ideas_recycled": total_recycled,
                "innovations_generated": processed,
                "duration_ms": duration_ms
            }),
        )
        .await;
    }

    /// Immediate trigger for cross-pollination (called by VQCIntegrator and others)
    pub async fn trigger_immediate_review() {
        // Lightweight path — still runs the full cascade under TOLC 8
        Self::run().await;
    }
}

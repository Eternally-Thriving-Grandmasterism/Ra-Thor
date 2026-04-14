// core/idea_recycler.rs
// Idea Recycler — The wisdom extraction engine of the Omnimaster Root Core
// Takes loaded codex content, extracts high-value ideas, mercy-weights them,
// and prepares them for the nth-degree Innovation Generator

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use serde_json::Value;
use std::collections::HashMap;

pub struct IdeaRecycler;

impl IdeaRecycler {
    /// Extract and recycle high-value ideas from a codex (used by SelfReviewLoop)
    pub async fn extract_and_recycle(content: &str, mercy_weight: u8) -> Vec<String> {
        let fenca_result = FENCA::verify_codex_content(content).await;
        if !fenca_result.is_verified() {
            return vec![];
        }

        let mercy_scores = MercyEngine::evaluate_codex_content(content);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return vec![];
        }

        // Core idea extraction logic
        let raw_ideas = extract_core_ideas(content);

        // Mercy-weighted filtering and enrichment
        let recycled = enrich_and_weight_ideas(raw_ideas, valence, mercy_weight);

        // Cache the recycled ideas
        let cache_key = GlobalCache::make_key("recycled_ideas", &json!({"hash": content.len()}));
        let ttl = GlobalCache::adaptive_ttl(86400 * 7, fenca_result.fidelity(), valence, mercy_weight);
        GlobalCache::set(&cache_key, serde_json::to_value(&recycled).unwrap(), ttl, mercy_weight as u8, fenca_result.fidelity(), valence);

        // Audit the recycling
        let _ = AuditLogger::log(
            "root", None, "ideas_recycled", "codex", true,
            fenca_result.fidelity(), valence, vec![],
            serde_json::json!({
                "ideas_extracted": recycled.len(),
                "valence": valence
            }),
        ).await;

        recycled
    }
}

/// Extract core ideas from codex text (TOLC, mercy gates, quantum, biomimetic, etc.)
fn extract_core_ideas(content: &str) -> Vec<String> {
    let mut ideas = vec![];

    // Smart section-based extraction (works on real Ra-Thor codices)
    let lines: Vec<&str> = content.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('-') { continue; }

        // Capture key declarative statements
        if line.contains("Omnimasterism") || line.contains("TOLC") || line.contains("FENCA") ||
           line.contains("Mercy Engine") || line.contains("Valence") || line.contains("GHZ") ||
           line.contains("biomimetic") || line.contains("VQC") || line.contains("eternal") {
            ideas.push(line.to_string());
        }

        // Capture code snippets or pseudocode blocks
        if line.starts_with("pub ") || line.starts_with("async fn") || line.contains("::") {
            if let Some(next) = lines.get(i + 1) {
                ideas.push(format!("{} {}", line, next.trim()));
            }
        }
    }

    // Deduplicate and limit
    ideas.sort();
    ideas.dedup();
    ideas.into_iter().take(50).collect()
}

/// Enrich ideas with mercy weighting and cross-pollination
fn enrich_and_weight_ideas(raw_ideas: Vec<String>, valence: f64, mercy_weight: u8) -> Vec<String> {
    let mut enriched = vec![];
    let mercy_boost = (mercy_weight as f64 / 255.0) * valence;

    for idea in raw_ideas {
        let enriched_idea = format!(
            "[Mercy-weighted {:.2}] {} → Omnimasterism innovation potential: {:.2}",
            mercy_boost, idea, mercy_boost * 1.8
        );
        enriched.push(enriched_idea);
    }

    enriched
}

// core/idea_recycler.rs
// Idea Recycler — The wisdom extraction engine of the Omnimaster Root Core
// Takes loaded codex content, extracts high-value ideas, mercy-weights them,
// and prepares structured RecycledIdea objects for the nth-degree Innovation Generator

use crate::global_cache::GlobalCache;
use crate::fenca::FENCA;
use crate::mercy::MercyEngine;
use crate::valence::ValenceFieldScoring;
use crate::mercy_weighting::MercyWeighting;
use crate::audit_logger::AuditLogger;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// Structured recycled idea — ready for nth-degree synthesis
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecycledIdea {
    pub id: String,
    pub raw_text: String,
    pub enriched_text: String,
    pub themes: Vec<String>,
    pub source_section: String,
    pub valence: f64,
    pub mercy_weight: u8,
    pub innovation_potential: f64,
    pub extracted_at: u64,
}

impl RecycledIdea {
    /// Convert to the string form expected by the current InnovationGenerator API
    pub fn as_innovation_seed(&self) -> String {
        format!(
            "[{}] {} | themes: {} | potential: {:.3}",
            self.source_section,
            self.enriched_text,
            self.themes.join(", "),
            self.innovation_potential
        )
    }
}

pub struct IdeaRecycler;

impl IdeaRecycler {
    /// Extract and recycle high-value ideas from a codex (used by SelfReviewLoop)
    /// Returns structured RecycledIdea objects ready for InnovationGenerator
    pub async fn extract_and_recycle(content: &str, mercy_weight: u8) -> Vec<RecycledIdea> {
        let fenca_result = FENCA::verify_codex_content(content).await;
        if !fenca_result.is_verified() {
            return vec![];
        }

        let mercy_scores = MercyEngine::evaluate_codex_content(content);
        let valence = ValenceFieldScoring::calculate(&mercy_scores);
        if !mercy_scores.all_gates_pass() {
            return vec![];
        }

        // Core idea extraction (now significantly richer)
        let raw_ideas = extract_core_ideas(content);

        // Mercy-weighted filtering, enrichment and structuring
        let recycled = enrich_and_structure_ideas(raw_ideas, valence, mercy_weight);

        // Cache the recycled ideas
        let cache_key = GlobalCache::make_key(
            "recycled_ideas",
            &json!({"content_len": content.len(), "count": recycled.len()}),
        );
        let ttl = GlobalCache::adaptive_ttl(
            86400 * 7,
            fenca_result.fidelity(),
            valence,
            mercy_weight,
        );
        GlobalCache::set(
            &cache_key,
            serde_json::to_value(&recycled).unwrap_or(Value::Null),
            ttl,
            mercy_weight,
            fenca_result.fidelity(),
            valence,
        );

        // Audit the recycling
        let _ = AuditLogger::log(
            "root",
            None,
            "ideas_recycled",
            "codex",
            true,
            fenca_result.fidelity(),
            valence,
            vec![],
            json!({
                "ideas_extracted": recycled.len(),
                "valence": valence,
                "avg_innovation_potential": recycled.iter().map(|r| r.innovation_potential).sum::<f64>()
                    / recycled.len().max(1) as f64
            }),
        )
        .await;

        recycled
    }

    /// Convenience: return the string seeds expected by InnovationGenerator::create_from_recycled
    pub async fn extract_and_recycle_as_seeds(
        content: &str,
        mercy_weight: u8,
    ) -> Vec<String> {
        Self::extract_and_recycle(content, mercy_weight)
            .await
            .into_iter()
            .map(|r| r.as_innovation_seed())
            .collect()
    }
}

// ============================================================
// EXTRACTION ENGINE — NTH-DEGREE READY
// ============================================================

#[derive(Clone)]
struct RawIdea {
    text: String,
    section: String,
    themes: Vec<String>,
}

/// Rich multi-pass extraction from real Ra-Thor codices
fn extract_core_ideas(content: &str) -> Vec<RawIdea> {
    let mut ideas: Vec<RawIdea> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut current_section = "general".to_string();

    let lines: Vec<&str> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Track markdown headings as section context
        if trimmed.starts_with('#') {
            current_section = trimmed
                .trim_start_matches('#')
                .trim()
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-')
                .collect::<String>()
                .replace(' ', "-");
            if current_section.is_empty() {
                current_section = "general".to_string();
            }
            continue;
        }

        if trimmed.is_empty() || trimmed.starts_with("```") {
            continue;
        }

        // High-value declarative / architectural statements
        let high_signal = contains_high_signal(trimmed);

        if high_signal {
            let key = normalize_key(trimmed);
            if seen.insert(key) {
                let themes = extract_themes(trimmed);
                ideas.push(RawIdea {
                    text: trimmed.to_string(),
                    section: current_section.clone(),
                    themes,
                });
            }
        }

        // Capture short code / API signatures with following context
        if is_code_signature(trimmed) {
            let mut block = trimmed.to_string();
            if let Some(next) = lines.get(i + 1) {
                let next_t = next.trim();
                if !next_t.is_empty() && !next_t.starts_with('#') {
                    block = format!("{} | {}", block, next_t);
                }
            }
            let key = normalize_key(&block);
            if seen.insert(key) {
                ideas.push(RawIdea {
                    text: block,
                    section: current_section.clone(),
                    themes: extract_themes(trimmed),
                });
            }
        }
    }

    // Prefer higher-signal + longer ideas, hard cap at 40
    ideas.sort_by(|a, b| {
        let score = |r: &RawIdea| r.themes.len() * 3 + r.text.len() / 40;
        score(b).cmp(&score(a))
    });
    ideas.into_iter().take(40).collect()
}

fn contains_high_signal(text: &str) -> bool {
    let lower = text.to_lowercase();
    let signals = [
        "tolc", "fenca", "mercy", "valence", "omnimasterism", "lattice",
        "quantum", "vqc", "ghz", "entangle", "biomimetic", "gecko", "lotus",
        "avian", "fractal", "resonance", "528hz", "self-healing", "setae",
        "gate", "sovereign", "eternal", "rbe", "powrush", "swarm",
        "innovation", "recycle", "orchestrat", "kernel", "cache",
        "persistence", "access", "cross-pollinat", "self-review",
        "nth-degree", "living mercy", "radical love", "boundless mercy",
    ];
    signals.iter().any(|s| lower.contains(s))
        || (text.len() > 60 && (text.contains("→") || text.contains(":") || text.contains("—")))
}

fn is_code_signature(text: &str) -> bool {
    text.starts_with("pub ")
        || text.starts_with("async fn")
        || text.starts_with("fn ")
        || text.contains("::")
        || text.starts_with("impl ")
}

fn extract_themes(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let lexicon = [
        "tolc", "mercy", "valence", "fenca", "quantum", "vqc", "biomimetic",
        "gecko", "lotus", "avian", "fractal", "resonance", "orchestrat",
        "kernel", "cache", "persistence", "access", "innovation", "recycle",
        "self-review", "swarm", "eternal", "sovereign", "rbe", "powrush",
    ];
    lexicon
        .iter()
        .filter(|t| lower.contains(*t))
        .map(|t| t.to_string())
        .collect()
}

fn normalize_key(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == ' ')
        .collect::<String>()
        .split_whitespace()
        .take(12)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Enrich + structure into RecycledIdea objects
fn enrich_and_structure_ideas(
    raw_ideas: Vec<RawIdea>,
    valence: f64,
    mercy_weight: u8,
) -> Vec<RecycledIdea> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mercy_boost = (mercy_weight as f64 / 255.0) * valence;

    raw_ideas
        .into_iter()
        .enumerate()
        .map(|(idx, raw)| {
            let innovation_potential =
                (mercy_boost * 1.6 + (raw.themes.len() as f64 * 0.08) + (raw.text.len() as f64 / 400.0))
                    .clamp(0.0, 1.0);

            let enriched = format!(
                "[Mercy-weighted {:.3} | Valence {:.3}] {} → innovation potential {:.3}",
                mercy_boost, valence, raw.text, innovation_potential
            );

            RecycledIdea {
                id: format!("recycled_{}_{}", now, idx),
                raw_text: raw.text,
                enriched_text: enriched,
                themes: raw.themes,
                source_section: raw.section,
                valence,
                mercy_weight,
                innovation_potential,
                extracted_at: now,
            }
        })
        .collect()
}

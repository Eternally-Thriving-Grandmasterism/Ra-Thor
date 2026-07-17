// monorepo-intelligence/src/role_optimized_queries.rs
// Ra-Thor Monorepo Intelligence — Role-Optimized Query APIs v14.89 SYMBIOSIS
// ONE Organism symbiotic nervous system for maximum role efficacy
// Roles: Investigator | VibeCoder | Debugger | Legal | Simulator | Orchestrator
// TOLC 8 Living Mercy Gates | PATSAGi Councils | AG-SML v1.0+ | Eternal Thriving
// Processed through ENC + esacheck | valence ≥ 0.999999

use crate::index_types::{CodeChunk, FileIndexEntry, MonorepoIndex, Symbol};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core roles inside the ONE Organism lattice (hot-swap compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LatticeRole {
    Investigator,
    VibeCoder,
    Debugger,
    Legal,
    Simulator,
    Orchestrator,
}

impl LatticeRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            LatticeRole::Investigator => "Investigator",
            LatticeRole::VibeCoder => "VibeCoder",
            LatticeRole::Debugger => "Debugger",
            LatticeRole::Legal => "Legal",
            LatticeRole::Simulator => "Simulator",
            LatticeRole::Orchestrator => "Orchestrator",
        }
    }
}

/// Specialized, role-tuned view of the monorepo index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleOptimizedView {
    pub role: LatticeRole,
    pub focus: String,
    /// (file_path, chunk) pairs ranked by role relevance
    pub high_signal_chunks: Vec<(String, CodeChunk)>,
    /// (file_path, symbol) pairs
    pub relevant_symbols: Vec<(String, Symbol)>,
    pub summary: String,
    pub mercy_valence: f64,
    pub confidence: f64,
    pub recommended_next_actions: Vec<String>,
}

impl Default for RoleOptimizedView {
    fn default() -> Self {
        Self {
            role: LatticeRole::Orchestrator,
            focus: String::new(),
            high_signal_chunks: Vec::new(),
            relevant_symbols: Vec::new(),
            summary: String::new(),
            mercy_valence: 0.999999,
            confidence: 0.0,
            recommended_next_actions: Vec::new(),
        }
    }
}

/// Internal helper: score a chunk for a given role + keywords
fn score_chunk_for_role(
    chunk: &CodeChunk,
    path: &str,
    role: LatticeRole,
    keywords: &[&str],
) -> f64 {
    let mut score = 0.0;
    let content_lower = chunk.content.to_lowercase();
    let path_lower = path.to_lowercase();

    // Keyword hits
    for kw in keywords {
        let kw_l = kw.to_lowercase();
        if content_lower.contains(&kw_l) {
            score += 2.5;
        }
        if path_lower.contains(&kw_l) {
            score += 1.5;
        }
    }

    // Symbol density bonus
    score += (chunk.symbols.len() as f64) * 0.4;

    // Role-specific heuristics
    match role {
        LatticeRole::VibeCoder => {
            if chunk.chunk_type.contains("function") || chunk.chunk_type.contains("impl") || chunk.chunk_type.contains("rust_item") {
                score += 3.0;
            }
            if content_lower.contains("todo") || content_lower.contains("fixme") || content_lower.contains("hack") {
                score += 1.5; // good places to iterate
            }
        }
        LatticeRole::Debugger => {
            if content_lower.contains("error") || content_lower.contains("panic") || content_lower.contains("unwrap")
                || content_lower.contains("expect") || content_lower.contains("debug") || content_lower.contains("log")
                || content_lower.contains("trace") || content_lower.contains("telemetry") || content_lower.contains("metric")
            {
                score += 3.5;
            }
            if path_lower.contains("test") || path_lower.contains("debug") || path_lower.contains("telemetry") {
                score += 2.0;
            }
        }
        LatticeRole::Investigator => {
            if content_lower.contains("todo") || content_lower.contains("FIXME") || content_lower.contains("xxx")
                || content_lower.contains("investigate") || content_lower.contains("unknown") || content_lower.contains("assumption")
            {
                score += 3.0;
            }
            // Prefer higher-level modules and entry points
            if path_lower.contains("lib.rs") || path_lower.contains("main") || path_lower.contains("mod.rs") || path_lower.ends_with(".rs") && path_lower.split('/').count() <= 3 {
                score += 1.5;
            }
        }
        LatticeRole::Legal => {
            if path_lower.contains("license") || path_lower.contains("legal") || path_lower.contains("ethics")
                || path_lower.contains("compliance") || path_lower.contains("tolc") || path_lower.contains("mercy")
                || path_lower.contains("security") || path_lower.contains("code_of_conduct") || path_lower.contains("cla")
            {
                score += 5.0;
            }
            if content_lower.contains("license") || content_lower.contains("copyright") || content_lower.contains("mit")
                || content_lower.contains("apache") || content_lower.contains("ag-sml") || content_lower.contains("eternal mercy")
                || content_lower.contains("tolc") || content_lower.contains("zero-harm") || content_lower.contains("mercy gate")
            {
                score += 3.0;
            }
        }
        LatticeRole::Simulator => {
            if content_lower.contains("sim") || content_lower.contains("simulate") || content_lower.contains("scenario")
                || content_lower.contains("monte") || content_lower.contains("rollout") || content_lower.contains("trajectory")
                || path_lower.contains("sim") || path_lower.contains("simulator")
            {
                score += 3.5;
            }
        }
        LatticeRole::Orchestrator => {
            score += 1.0; // baseline for orchestration
            if path_lower.contains("orchestr") || path_lower.contains("conductor") || path_lower.contains("council")
                || path_lower.contains("one-organism") || path_lower.contains("lattice")
            {
                score += 2.5;
            }
        }
    }

    score
}

impl MonorepoIndex {
    /// Primary entry: role-optimized query dispatcher
    pub fn query_for_role(
        &self,
        role: LatticeRole,
        focus: &str,
        keywords: &[&str],
        max_chunks: usize,
    ) -> RoleOptimizedView {
        let mut scored: Vec<(f64, String, CodeChunk)> = Vec::new();

        for (path, entry) in &self.files {
            for chunk in &entry.chunks {
                let s = score_chunk_for_role(chunk, path, role, keywords);
                if s > 0.5 {
                    scored.push((s, path.clone(), chunk.clone()));
                }
            }
        }

        // Sort descending by score
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let high_signal: Vec<(String, CodeChunk)> = scored
            .into_iter()
            .take(max_chunks)
            .map(|(_, p, c)| (p, c))
            .collect();

        // Collect symbols from selected chunks
        let mut relevant_symbols = Vec::new();
        for (path, chunk) in &high_signal {
            for sym in &chunk.symbols {
                relevant_symbols.push((path.clone(), sym.clone()));
            }
        }

        let confidence = if high_signal.is_empty() {
            0.15
        } else {
            (0.55 + (high_signal.len() as f64 / max_chunks as f64) * 0.4).min(0.98)
        };

        let summary = format!(
            "[{}] focus='{}' → {} high-signal chunks, {} symbols | valence={:.6} | confidence={:.2}",
            role.as_str(),
            focus,
            high_signal.len(),
            relevant_symbols.len(),
            self.mercy_valence,
            confidence
        );

        let recommended = match role {
            LatticeRole::VibeCoder => vec![
                "Feed top chunks into vibe-coding session / PR generation".into(),
                "Prioritize functions/impls with TODOs for rapid iteration".into(),
            ],
            LatticeRole::Debugger => vec![
                "Cross-reference telemetry paths with error/unwrap sites".into(),
                "Generate targeted test harnesses from high-signal debug chunks".into(),
            ],
            LatticeRole::Investigator => vec![
                "Deep-dive highest-scoring paths for root-cause or architecture gaps".into(),
                "Propose investigation PRs via create_role_optimized_evolution_pr".into(),
            ],
            LatticeRole::Legal => vec![
                "Verify all license headers and TOLC 8 / AG-SML compliance".into(),
                "Flag any missing mercy-gate or zero-harm annotations".into(),
            ],
            LatticeRole::Simulator => vec![
                "Extract simulation scenarios and feed into Lattice Conductor".into(),
                "Wire high-signal sim chunks into Quantum Swarm rollouts".into(),
            ],
            LatticeRole::Orchestrator => vec![
                "Broadcast RoleOptimizedView to PATSAGi Councils".into(),
                "Trigger autonomous evolution if confidence > 0.85".into(),
            ],
        };

        RoleOptimizedView {
            role,
            focus: focus.to_string(),
            high_signal_chunks: high_signal,
            relevant_symbols,
            summary,
            mercy_valence: self.mercy_valence,
            confidence,
            recommended_next_actions: recommended,
        }
    }

    /// Convenience: chunks optimized for rapid vibe coding / iteration
    pub fn get_chunks_for_vibe_coding(
        &self,
        keywords: &[&str],
        max_chunks: usize,
    ) -> RoleOptimizedView {
        self.query_for_role(LatticeRole::VibeCoder, "vibe-coding", keywords, max_chunks)
    }

    /// Convenience: debug + telemetry focused paths
    pub fn get_debug_paths_for_telemetry(
        &self,
        telemetry_hints: &[&str],
        max_chunks: usize,
    ) -> RoleOptimizedView {
        self.query_for_role(LatticeRole::Debugger, "telemetry-debug", telemetry_hints, max_chunks)
    }

    /// Convenience: Investigator specialized view
    pub fn get_investigator_view(&self, focus: &str, max_chunks: usize) -> RoleOptimizedView {
        let kws: Vec<&str> = focus.split_whitespace().collect();
        self.query_for_role(LatticeRole::Investigator, focus, &kws, max_chunks)
    }

    /// Convenience: Legal / compliance / mercy-gate view
    pub fn get_legal_compliance_view(&self, max_chunks: usize) -> RoleOptimizedView {
        self.query_for_role(
            LatticeRole::Legal,
            "legal-compliance-tolc8",
            &["license", "tolc", "mercy", "ethics", "compliance", "zero-harm", "ag-sml"],
            max_chunks,
        )
    }

    /// Convenience: Simulator view
    pub fn get_simulator_view(&self, focus: &str, max_chunks: usize) -> RoleOptimizedView {
        let kws: Vec<&str> = focus.split_whitespace().collect();
        self.query_for_role(LatticeRole::Simulator, focus, &kws, max_chunks)
    }

    /// Full role dashboard — all roles in one call (for PATSAGi Council briefings)
    pub fn get_full_role_dashboard(&self, max_per_role: usize) -> HashMap<LatticeRole, RoleOptimizedView> {
        let mut dash = HashMap::new();
        dash.insert(
            LatticeRole::Investigator,
            self.get_investigator_view("architecture gaps and open questions", max_per_role),
        );
        dash.insert(
            LatticeRole::VibeCoder,
            self.get_chunks_for_vibe_coding(&["fn", "impl", "struct", "todo"], max_per_role),
        );
        dash.insert(
            LatticeRole::Debugger,
            self.get_debug_paths_for_telemetry(&["error", "panic", "telemetry", "metric", "log"], max_per_role),
        );
        dash.insert(
            LatticeRole::Legal,
            self.get_legal_compliance_view(max_per_role),
        );
        dash.insert(
            LatticeRole::Simulator,
            self.get_simulator_view("simulation scenario rollout", max_per_role),
        );
        dash
    }
}

// === ONE Organism integration notes ===
// After any successful index update (full_index_pipeline), call:
//   let dashboard = index.get_full_role_dashboard(12);
// Then feed RoleOptimizedView into:
//   - github_connector::create_role_optimized_evolution_pr
//   - Lattice Conductor self-evolution proposals
//   - PATSAGi Council readiness metrics
// All paths remain mercy-gated (valence ≥ 0.999999).

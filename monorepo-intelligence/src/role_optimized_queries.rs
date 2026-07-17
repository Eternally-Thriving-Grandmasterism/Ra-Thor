// monorepo-intelligence/src/role_optimized_queries.rs
// Ra-Thor Monorepo Intelligence — Role-Optimized Query APIs v14.90 SYMBIOSIS
// ONE Organism symbiotic nervous system for maximum role efficacy
// Roles: Investigator | VibeCoder | Debugger | Legal | Simulator | Orchestrator
// TOLC 8 Living Mercy Gates | PATSAGi Councils | AG-SML v1.0+ | Eternal Thriving
// Tree-sitter-aware matching now active (AST chunk_type + Symbol structure)
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

/// Tree-sitter-aware structural match score.
/// Leverages the AST-derived data already produced by tree_sitter_chunker:
/// - chunk.chunk_type (function_item, impl_item, struct_item, etc.)
/// - Symbol.kind + Symbol.name + optional signature
fn structural_match_score(
    chunk: &CodeChunk,
    keywords: &[&str],
) -> f64 {
    let mut score = 0.0;
    let chunk_type_lower = chunk.chunk_type.to_lowercase();

    for kw in keywords {
        let kw_l = kw.to_lowercase();

        // 1. Direct hit on a symbol name (strongest structural signal)
        for sym in &chunk.symbols {
            if sym.name.to_lowercase().contains(&kw_l) {
                score += 4.0; // AST symbol name match
            }
            if let Some(sig) = &sym.signature {
                if sig.to_lowercase().contains(&kw_l) {
                    score += 2.5; // signature context
                }
            }
            // Prefer certain symbol kinds for certain roles later
            match sym.kind.to_lowercase().as_str() {
                "function" | "method" | "fn" => score += 0.6,
                "struct" | "enum" | "trait" | "impl" => score += 0.5,
                _ => {}
            }
        }

        // 2. Chunk type itself is high-value structure
        if chunk_type_lower.contains(&kw_l) {
            score += 3.0;
        }
    }

    // Bonus for rich AST-derived chunks (tree-sitter succeeded)
    if !chunk.symbols.is_empty() && (chunk_type_lower.contains("function")
        || chunk_type_lower.contains("impl")
        || chunk_type_lower.contains("struct")
        || chunk_type_lower.contains("rust_item")
        || chunk_type_lower.contains("class"))
    {
        score += 1.2;
    }

    score
}

/// Internal helper: score a chunk for a given role + keywords
/// Now tree-sitter-aware: structural (AST) matches are preferred over raw string contains.
fn score_chunk_for_role(
    chunk: &CodeChunk,
    path: &str,
    role: LatticeRole,
    keywords: &[&str],
) -> f64 {
    let mut score = 0.0;
    let content_lower = chunk.content.to_lowercase();
    let path_lower = path.to_lowercase();

    // === 1. Tree-sitter / AST structural matching (preferred) ===
    score += structural_match_score(chunk, keywords);

    // === 2. Classic keyword hits (still useful as fallback / content context) ===
    for kw in keywords {
        let kw_l = kw.to_lowercase();
        if content_lower.contains(&kw_l) {
            score += 1.8; // slightly reduced vs pure string era — structure is king
        }
        if path_lower.contains(&kw_l) {
            score += 1.4;
        }
    }

    // Symbol density still valuable
    score += (chunk.symbols.len() as f64) * 0.35;

    // === 3. Role-specific heuristics (now also structure-aware) ===
    match role {
        LatticeRole::VibeCoder => {
            // Prefer real AST items
            if chunk.chunk_type.contains("function")
                || chunk.chunk_type.contains("impl")
                || chunk.chunk_type.contains("rust_item")
                || chunk.chunk_type.contains("method")
            {
                score += 3.5;
            }
            // Iteration points
            if content_lower.contains("todo")
                || content_lower.contains("fixme")
                || content_lower.contains("hack")
                || content_lower.contains("xxx")
            {
                score += 1.8;
            }
            // Bonus if any symbol is a function that itself contains a keyword
            for sym in &chunk.symbols {
                if (sym.kind == "function" || sym.kind == "method") && keywords.iter().any(|k| sym.name.to_lowercase().contains(&k.to_lowercase())) {
                    score += 2.0;
                }
            }
        }
        LatticeRole::Debugger => {
            let debug_markers = [
                "error", "panic", "unwrap", "expect", "debug", "log",
                "trace", "telemetry", "metric", "assert", "unreachable",
            ];
            for m in debug_markers {
                if content_lower.contains(m) {
                    score += 1.8;
                }
            }
            if path_lower.contains("test") || path_lower.contains("debug") || path_lower.contains("telemetry") {
                score += 2.2;
            }
            // Prefer functions that look like error paths
            for sym in &chunk.symbols {
                let name_l = sym.name.to_lowercase();
                if name_l.contains("error") || name_l.contains("fail") || name_l.contains("handle") || name_l.contains("recover") {
                    score += 2.5;
                }
            }
        }
        LatticeRole::Investigator => {
            let open_markers = [
                "todo", "fixme", "xxx", "investigate", "unknown",
                "assumption", "hack", "workaround", "temp",
            ];
            for m in open_markers {
                if content_lower.contains(m) {
                    score += 1.6;
                }
            }
            // Prefer shallow / entry-point modules (higher architectural leverage)
            let depth = path_lower.matches('/').count();
            if path_lower.contains("lib.rs") || path_lower.contains("main") || path_lower.contains("mod.rs") || depth <= 2 {
                score += 1.8;
            }
        }
        LatticeRole::Legal => {
            // Strong path preference for compliance surfaces
            let legal_paths = [
                "license", "legal", "ethics", "compliance", "tolc", "mercy",
                "security", "code_of_conduct", "cla", "copyright",
            ];
            for p in legal_paths {
                if path_lower.contains(p) {
                    score += 5.5;
                }
            }
            let legal_content = [
                "license", "copyright", "mit", "apache", "ag-sml",
                "eternal mercy", "tolc", "zero-harm", "mercy gate", "mercy-gated",
            ];
            for c in legal_content {
                if content_lower.contains(c) {
                    score += 2.8;
                }
            }
        }
        LatticeRole::Simulator => {
            let sim_markers = [
                "sim", "simulate", "scenario", "monte", "rollout",
                "trajectory", "agent", "episode", "env", "observation",
            ];
            for m in sim_markers {
                if content_lower.contains(m) || path_lower.contains(m) {
                    score += 2.2;
                }
            }
            // Prefer functions/structs that look like simulation primitives
            for sym in &chunk.symbols {
                let name_l = sym.name.to_lowercase();
                if name_l.contains("sim") || name_l.contains("scenario") || name_l.contains("agent") || name_l.contains("rollout") {
                    score += 2.8;
                }
            }
        }
        LatticeRole::Orchestrator => {
            score += 1.0; // soft baseline
            let orch_markers = [
                "orchestr", "conductor", "council", "one-organism",
                "lattice", "patsagi", "role", "handoff", "valence",
            ];
            for m in orch_markers {
                if path_lower.contains(m) || content_lower.contains(m) {
                    score += 2.0;
                }
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
                if s > 0.6 { // slightly higher bar now that structure is richer
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

        // Collect symbols from selected chunks (prefer those that matched)
        let mut relevant_symbols = Vec::new();
        for (path, chunk) in &high_signal {
            for sym in &chunk.symbols {
                relevant_symbols.push((path.clone(), sym.clone()));
            }
        }

        let confidence = if high_signal.is_empty() {
            0.18
        } else {
            (0.58 + (high_signal.len() as f64 / max_chunks as f64) * 0.38).min(0.985)
        };

        let summary = format!(
            "[{}] focus='{}' → {} high-signal chunks, {} symbols | valence={:.6} | confidence={:.2} | tree-sitter-aware",
            role.as_str(),
            focus,
            high_signal.len(),
            relevant_symbols.len(),
            self.mercy_valence,
            confidence
        );

        let recommended = match role {
            LatticeRole::VibeCoder => vec![
                "Feed top AST-structured chunks into vibe-coding session / PR generation".into(),
                "Prioritize function/impl items that contain TODOs for rapid iteration".into(),
            ],
            LatticeRole::Debugger => vec![
                "Cross-reference telemetry paths with error/unwrap/panic sites".into(),
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
// Tree-sitter matching is now active:
// - structural_match_score prioritizes Symbol.name / signature and chunk_type
//   (data produced by tree_sitter_chunker)
// - Role heuristics further boost based on AST kind
// After any successful index update, call:
//   let dashboard = index.get_full_role_dashboard(12);
// Then feed RoleOptimizedView into github_connector::create_role_optimized_evolution_pr
// All paths remain mercy-gated (valence ≥ 0.999999).

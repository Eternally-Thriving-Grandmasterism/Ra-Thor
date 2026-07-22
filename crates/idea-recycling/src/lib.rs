//! Idea Recycling System — Core wisdom extraction and mercy-weighted cross-pollination engine
//! Part of the Self-Evolution Looping Systems for Ra-Thor / Rathor.ai
//!
//! This crate provides a clean, reusable surface over the authoritative logic that lives in
//! `core/idea_recycler.rs`. Prefer the core implementation inside the monorepo Root Core;
//! this crate exists for external / modular consumers and lattice-wide imports.

use serde::{Deserialize, Serialize};
use tracing::info;

/// Structured recycled idea — mirror of the core type for crate consumers
#[derive(Clone, Debug, Serialize, Deserialize)]
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

/// Main Idea Recycling engine (crate-level surface)
pub struct IdeaRecycler;

impl IdeaRecycler {
    pub fn new() -> Self {
        Self
    }

    /// Recycle a single idea string with mercy-gating awareness
    pub async fn recycle_idea(&self, idea: &str) -> Result<String, String> {
        info!("Recycling single idea with mercy-gating and FENCA awareness...");
        // Lightweight enrichment for external callers
        let enriched = format!(
            "[Mercy-weighted seed] {} → ready for nth-degree Innovation Generator",
            idea
        );
        Ok(enriched)
    }

    /// Cross-pollinate a set of ideas (simple lattice-ready pass)
    pub async fn cross_pollinate(&self, ideas: Vec<String>) -> Result<Vec<String>, String> {
        info!("Cross-pollinating {} ideas across the lattice...", ideas.len());
        let pollinated: Vec<String> = ideas
            .into_iter()
            .map(|idea| format!("[Cross-pollinated] {}", idea))
            .collect();
        Ok(pollinated)
    }

    /// Convert structured ideas into InnovationGenerator seeds
    pub fn to_seeds(ideas: &[RecycledIdea]) -> Vec<String> {
        ideas.iter().map(|r| r.as_innovation_seed()).collect()
    }
}

impl Default for IdeaRecycler {
    fn default() -> Self {
        Self::new()
    }
}

pub fn init_idea_recycling() {
    info!("Idea Recycling System (crate) initialized and ready for Self-Evolution Looping Systems");
}

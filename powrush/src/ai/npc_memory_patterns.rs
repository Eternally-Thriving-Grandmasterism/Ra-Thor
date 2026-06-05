//! NPC Memory Patterns for Powrush-MMO + Ra-Thor AGI (v16.0 Production)
//!
//! Advanced, persistent, vector-enhanced memory system for NPCs.
//! Designed to work with the SurrealDB cluster, vector search, and live queries
//! to create intelligent, context-aware, mercy-aligned NPCs.
//!
//! Memory Types:
//! - Episodic: Specific events and interactions with players
//! - Semantic: Generalized knowledge (via vector similarity)
//! - Emotional/Alignment: Mercy, volatility, and relationship tracking
//!
//! This enables NPCs to:
//! - Remember individual players across sessions
//! - Recognize similar player archetypes using vector search
//! - React intelligently to world changes via live queries
//! - Contribute to meaningful player learning and RBE experiences
//!
//! Integrates with PATSAGi Councils and Ra-Thor for high-level decision making.
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::persistence::vector_search::{PlayerVector, generate_player_embedding};
use crate::systems::epigenetic_modulation::EpigeneticProfile;

/// Types of memory an NPC can hold.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryType {
    Episodic,      // Specific interaction with a player
    Semantic,      // Generalized knowledge (e.g., "Players who help others tend to have high mercy_alignment")
    Relationship,  // Long-term feeling toward a specific player or faction
    WorldEvent,    // Major world changes (layer advances, major conflicts)
}

/// A single memory entry stored by an NPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcMemoryEntry {
    pub memory_type: MemoryType,
    pub timestamp: u64,
    pub player_id: Option<u64>,
    pub region_id: Option<u64>,
    pub content: String,                    // Human-readable or structured description
    pub embedding: Option<Vec<f32>>,        // Vector for semantic similarity
    pub emotional_valence: f32,             // -1.0 (negative) to +1.0 (positive) — mercy-aligned
    pub importance: f32,                    // How strongly the NPC remembers this
}

/// Per-NPC or shared memory store.
#[derive(Resource, Clone, Default)]
pub struct NpcMemoryStore {
    /// npc_id -> list of memories
    pub memories: HashMap<u64, Vec<NpcMemoryEntry>>,
    /// Quick lookup for relationship strength with players
    pub player_relationships: HashMap<(u64, u64), f32>, // (npc_id, player_id) -> relationship score
}

impl NpcMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new memory (episodic or relationship-based).
    pub fn record_memory(&mut self, npc_id: u64, entry: NpcMemoryEntry) {
        self.memories.entry(npc_id).or_default().push(entry.clone());

        if let (Some(player_id), MemoryType::Relationship) = (entry.player_id, &entry.memory_type) {
            let key = (npc_id, player_id);
            let current = self.player_relationships.get(&key).copied().unwrap_or(0.0);
            self.player_relationships.insert(key, (current + entry.emotional_valence).clamp(-1.0, 1.0));
        }
    }

    /// Retrieve memories for an NPC, optionally filtered by type.
    pub fn get_memories(&self, npc_id: u64, filter: Option<MemoryType>) -> Vec<&NpcMemoryEntry> {
        if let Some(memories) = self.memories.get(&npc_id) {
            if let Some(ft) = filter {
                return memories.iter().filter(|m| m.memory_type == ft).collect();
            } else {
                return memories.iter().collect();
            }
        }
        vec![]
    }

    /// Find similar past experiences using vector similarity (future: powered by SurrealDB vector search).
    pub fn find_similar_memories(
        &self,
        npc_id: u64,
        query_embedding: &[f32],
        limit: usize,
    ) -> Vec<&NpcMemoryEntry> {
        // Placeholder for vector similarity search
        // In production: Use SurrealDB vector search on stored embeddings
        if let Some(memories) = self.memories.get(&npc_id) {
            let mut scored: Vec<_> = memories
                .iter()
                .filter_map(|m| {
                    m.embedding.as_ref().map(|emb| {
                        let similarity = cosine_similarity(query_embedding, emb);
                        (similarity, m)
                    })
                })
                .collect();

            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            return scored.into_iter().take(limit).map(|(_, m)| m).collect();
        }
        vec![]
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// System that allows NPCs to form new memories from player interactions or world events.
pub fn npc_memory_formation_system(
    mut memory_store: ResMut<NpcMemoryStore>,
    // In real system: Query for relevant events, player interactions, or live query changes
) {
    // Example: Periodically or on events, NPCs record memories
    // This would be triggered by gameplay systems or live query events
}

/// Example: NPC recalls similar past experiences when encountering a player.
pub fn npc_memory_recall_system(
    memory_store: Res<NpcMemoryStore>,
    // player_profile: &EpigeneticProfile,
) {
    // Generate embedding from current player
    // let query_vec = generate_player_embedding(player_profile);
    // let similar = memory_store.find_similar_memories(npc_id, &query_vec, 5);
    // Use similar memories to influence NPC behavior / dialogue / decisions
}

/// Persistence hook: Save important NPC memories to SurrealDB cluster.
/// This allows long-term memory across server restarts and supports Ra-Thor AGI analysis.
pub async fn persist_npc_memories(
    memory_store: &NpcMemoryStore,
    // persistence: &SurrealPersistence,
) {
    // In production: Batch insert important memories into a dedicated `npc_memory` table
    // with embeddings for vector search
}

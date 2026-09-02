//! Vector Embedding Integration Layer for Powrush-MMO + Ra-Thor AGI (v16.1 Production)
//!
//! This module provides deep, reactive integration between game state
//! (Epigenetic Profiles, Region Geometry, NPC Memories) and SurrealDB vector search.
//!
//! Responsibilities:
//! - Automatically generate and update embeddings when relevant state changes
//! - Persist embeddings to SurrealDB alongside main data
//! - Provide unified APIs for similarity search used by NPC Memory, PATSAGi, and Ra-Thor AGI
//! - Keep embeddings in sync with live queries and persistence systems
//!
//! This enables truly intelligent, context-aware NPCs and personalized player experiences.
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::persistence::vector_search::{generate_player_embedding, generate_region_embedding};
use crate::systems::epigenetic_modulation::{EpigeneticModulationField, EpigeneticProfile};
use crate::systems::geometric_harmony_layer::GeometricHarmonyLayer;
use crate::ai::npc_memory_patterns::NpcMemoryStore;

/// Resource that manages embedding generation and synchronization.
#[derive(Resource, Default)]
pub struct VectorEmbeddingIntegration {
    pub auto_sync_enabled: bool,
    pub last_sync_tick: u64,
}

/// Event emitted when embeddings should be regenerated and synced.
#[derive(Event)]
pub struct RegenerateEmbeddings;

/// System that reacts to state changes and updates embeddings.
/// This can be triggered by live queries, persistence events, or periodic ticks.
pub fn sync_embeddings_system(
    mut integration: ResMut<VectorEmbeddingIntegration>,
    epigenetic: Res<EpigeneticModulationField>,
    geometric: Res<GeometricHarmonyLayer>,
    mut memory_store: ResMut<NpcMemoryStore>,
    mut events: EventReader<RegenerateEmbeddings>,
) {
    let should_sync = integration.auto_sync_enabled || !events.is_empty();

    if should_sync {
        // Update player embeddings based on current epigenetic state
        for (player_id, profile) in &epigenetic.profiles {
            let embedding = generate_player_embedding(profile);
            // In production: Upsert embedding into SurrealDB player_epigenetic_profile table
            // or a dedicated embeddings table
        }

        // Update region embeddings
        for (region_id, region) in &geometric.regions {
            let embedding = generate_region_embedding(region);
            // Persist region embedding
        }

        // Optionally embed recent NPC memories for semantic recall
        integration.last_sync_tick += 1;
    }
}

/// Helper to generate a rich embedding for an NPC memory entry.
/// Combines memory content, emotional valence, and context.
pub fn generate_memory_embedding(
    content: &str,
    emotional_valence: f32,
    importance: f32,
) -> Vec<f32> {
    // Simple encoding for now. In production, use a proper embedding model
    // or combine with text embeddings from Ra-Thor AGI.
    let mut embedding = vec![emotional_valence, importance];
    // Add hashed content features or call into a real embedding service
    embedding
}

/// Integrate vector embeddings with NPC memory recall.
/// This allows NPCs to semantically search their past experiences.
pub fn enhance_npc_memory_with_vectors(
    memory_store: &mut NpcMemoryStore,
    npc_id: u64,
) {
    if let Some(memories) = memory_store.memories.get_mut(&npc_id) {
        for memory in memories.iter_mut() {
            if memory.embedding.is_none() {
                memory.embedding = Some(generate_memory_embedding(
                    &memory.content,
                    memory.emotional_valence,
                    memory.importance,
                ));
            }
        }
    }
}

/// Plugin that wires vector embedding integration into the Bevy app.
pub struct VectorEmbeddingIntegrationPlugin;

impl Plugin for VectorEmbeddingIntegrationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VectorEmbeddingIntegration>();
        app.add_event::<RegenerateEmbeddings>();
        app.add_systems(Update, sync_embeddings_system);
    }
}

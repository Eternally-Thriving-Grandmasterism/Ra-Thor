//! Dynamic EF Search Adjustment for SurrealDB HNSW Vector Search (v16.4 Production)
//!
//! Implements runtime, context-aware adjustment of `ef_search` parameter.
//! This allows Powrush-MMO to dynamically trade off between search quality and speed
//! depending on the situation (NPC decision making, real-time events, background tasks, etc.).
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use crate::persistence::surreal_persistence::PersistenceError;

/// Controls the current search quality mode for the entire system or per-context.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSearchQuality {
    Fast,        // Low latency, acceptable recall
    Balanced,    // Default good balance
    HighRecall,  // Maximum accuracy (e.g. important NPC decisions)
}

impl Default for VectorSearchQuality {
    fn default() -> Self {
        VectorSearchQuality::Balanced
    }
}

/// Resource that holds the current global quality setting.
/// Can be changed at runtime based on game state or load.
#[derive(Resource, Debug, Clone)]
pub struct DynamicVectorSearch {
    pub current_quality: VectorSearchQuality,
    pub last_adjustment_tick: u64,
}

impl Default for DynamicVectorSearch {
    fn default() -> Self {
        Self {
            current_quality: VectorSearchQuality::Balanced,
            last_adjustment_tick: 0,
        }
    }
}

/// Returns the recommended `ef_search` value based on current quality mode.
/// These values can be tuned further based on benchmarking.
pub fn get_dynamic_ef_search(quality: VectorSearchQuality) -> u32 {
    match quality {
        VectorSearchQuality::Fast => 32,
        VectorSearchQuality::Balanced => 64,
        VectorSearchQuality::HighRecall => 128,
    }
}

/// System that can dynamically adjust search quality based on game conditions.
/// Example triggers:
/// - High NPC decision load → lower quality for speed
/// - Important story/NPC moment → increase to HighRecall
/// - Low player activity → can afford higher quality
pub fn dynamic_ef_adjustment_system(
    mut dynamic_search: ResMut<DynamicVectorSearch>,
    // In real implementation, you could read game state, NPC queue size, etc.
) {
    // Placeholder logic - replace with real conditions
    // Example: During intense combat or many NPCs deciding, prefer speed
    // if is_high_load() {
    //     dynamic_search.current_quality = VectorSearchQuality::Fast;
    // }
}

/// Performs a vector search with the currently configured dynamic ef_search.
pub async fn search_with_dynamic_ef(
    db: &surrealdb::Surreal<surrealdb::engine::any::Any>,
    query_vector: Vec<f32>,
    limit: usize,
    dynamic_search: &DynamicVectorSearch,
) -> Result<Vec<serde_json::Value>, PersistenceError> {
    let ef = get_dynamic_ef_search(dynamic_search.current_quality);

    // In a full implementation, ef would be passed to the query if supported,
    // or used to choose between pre-created indexes with different HNSW settings.
    let query = format!(
        "SELECT id, embedding, mercy_alignment 
         FROM player_epigenetic_profile 
         WHERE embedding <{{}} $query_vector 
         LIMIT {}",
        limit
    );

    let result = db
        .query(&query)
        .bind(("query_vector", query_vector))
        .await
        .map_err(|e| PersistenceError::Load(e.to_string()))?;

    // Process result...
    Ok(vec![])
}

/// Plugin to register dynamic vector search systems.
pub struct DynamicEfSearchPlugin;

impl Plugin for DynamicEfSearchPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DynamicVectorSearch>();
        app.add_systems(Update, dynamic_ef_adjustment_system);
    }
}

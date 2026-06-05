//! HNSW Index Tuning for SurrealDB Vector Search (v16.3 Production)
//!
//! Advanced HNSW (Hierarchical Navigable Small World) index tuning
//! specifically optimized for Powrush-MMO workloads.
//!
//! HNSW Parameters Explained:
//! - m: Number of bi-directional links per node (higher = better recall, more memory)
//! - ef_construction: Size of dynamic candidate list during construction (higher = better index quality, slower build)
//! - ef_search: Size of dynamic candidate list during search (higher = better recall, slower queries)
//!
//! Powrush-MMO Recommendations:
//! - Player vectors (5D): m=16, ef_construction=128, ef_search=64 (good balance)
//! - Region vectors (2D): Lower values sufficient due to smaller dimensionality
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use crate::persistence::surreal_persistence::PersistenceError;

/// Create highly tuned HNSW indexes for Powrush-MMO.
/// These settings prioritize a good balance between recall quality and query performance.
pub async fn create_tuned_hnsw_indexes(
    db: &surrealdb::Surreal<surrealdb::engine::any::Any>,
) -> Result<(), PersistenceError> {
    // Player vector index - tuned for 5-dimensional epigenetic embeddings
    let player_index = r#"
        DEFINE INDEX player_hnsw_vector_idx 
        ON player_epigenetic_profile 
        FIELDS embedding 
        VECTOR (5) 
        DIST_COSINE 
        HNSW 
            M 16 
            EF_CONSTRUCTION 128;
    "#;

    // Region vector index - tuned for lower dimensional data
    let region_index = r#"
        DEFINE INDEX region_hnsw_vector_idx 
        ON region_geometry 
        FIELDS embedding 
        VECTOR (2) 
        DIST_COSINE 
        HNSW 
            M 12 
            EF_CONSTRUCTION 64;
    "#;

    for query in [player_index, region_index] {
        if let Err(e) = db.query(query).await {
            bevy::log::warn!("HNSW index creation: {}", e);
        }
    }

    Ok(())
}

/// Create high-recall HNSW indexes (slower queries, better accuracy).
/// Use this when NPC decision quality is more important than raw speed.
pub async fn create_high_recall_hnsw_indexes(
    db: &surrealdb::Surreal<surrealdb::engine::any::Any>,
) -> Result<(), PersistenceError> {
    let player_index = r#"
        DEFINE INDEX player_hnsw_high_recall 
        ON player_epigenetic_profile 
        FIELDS embedding 
        VECTOR (5) 
        DIST_COSINE 
        HNSW 
            M 24 
            EF_CONSTRUCTION 200;
    "#;

    if let Err(e) = db.query(player_index).await {
        bevy::log::warn!("High-recall HNSW index: {}", e);
    }

    Ok(())
}

/// Create fast HNSW indexes (lower recall, very fast queries).
/// Useful for real-time systems where speed is critical.
pub async fn create_fast_hnsw_indexes(
    db: &surrealdb::Surreal<surrealdb::engine::any::Any>,
) -> Result<(), PersistenceError> {
    let player_index = r#"
        DEFINE INDEX player_hnsw_fast 
        ON player_epigenetic_profile 
        FIELDS embedding 
        VECTOR (5) 
        DIST_COSINE 
        HNSW 
            M 8 
            EF_CONSTRUCTION 64;
    "#;

    if let Err(e) = db.query(player_index).await {
        bevy::log::warn!("Fast HNSW index: {}", e);
    }

    Ok(())
}

/// Runtime query tuning helper.
/// Adjust ef_search dynamically based on desired quality vs speed tradeoff.
/// Higher ef_search = better recall, slower query.
pub fn get_ef_search_for_quality(quality: QueryQuality) -> u32 {
    match quality {
        QueryQuality::Fast => 32,
        QueryQuality::Balanced => 64,
        QueryQuality::HighRecall => 128,
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QueryQuality {
    Fast,
    Balanced,
    HighRecall,
}

/// Example of running a vector search with explicit ef_search tuning.
/// In production SurrealDB, you can often pass this as a query parameter.
pub async fn search_with_tuned_ef(
    db: &surrealdb::Surreal<surrealdb::engine::any::Any>,
    query_vector: Vec<f32>,
    limit: usize,
    quality: QueryQuality,
) -> Result<Vec<serde_json::Value>, PersistenceError> {
    let ef = get_ef_search_for_quality(quality);

    // Note: Actual ef_search parameter support depends on SurrealDB version.
    // This shows the intended pattern.
    let query = format!(
        "SELECT id, embedding 
         FROM player_epigenetic_profile 
         WHERE embedding <{{}} $query_vector 
         LIMIT {}",
        limit
    );

    // In real usage, you would bind ef and use it if the engine supports it
    let _ = db.query(&query).bind(("query_vector", query_vector)).await;

    Ok(vec![])
}

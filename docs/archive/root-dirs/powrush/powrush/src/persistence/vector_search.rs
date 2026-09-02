//! Automatic Quality Scaling for SurrealDB Vector Search (v16.5 Production)
//!
//! Implements automatic, intelligent scaling of vector search quality
//! based on runtime system conditions.
//!
//! Features:
//! - Automatic downgrade to Fast mode under high load
//! - Upgrade to HighRecall when system has capacity
//! - Hysteresis to prevent rapid flipping between modes
//! - Integration with Bevy schedule and DynamicVectorSearch resource
//!
//! This ensures good performance while still delivering high-quality
//! NPC decisions and player experiences when possible.
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use crate::persistence::vector_search::{DynamicVectorSearch, VectorSearchQuality};

/// Resource to track metrics relevant to quality scaling decisions.
#[derive(Resource, Debug, Default)]
pub struct VectorSearchMetrics {
    pub pending_npc_decisions: u32,
    pub average_frame_time_ms: f32,
    pub active_players: u32,
    pub last_scale_tick: u64,
}

/// System that automatically adjusts vector search quality.
/// This runs every frame or on a fixed timestep.
pub fn automatic_quality_scaling_system(
    mut dynamic_search: ResMut<DynamicVectorSearch>,
    mut metrics: ResMut<VectorSearchMetrics>,
) {
    let current = dynamic_search.current_quality;
    let new_quality = determine_optimal_quality(&metrics, current);

    if new_quality != current {
        dynamic_search.current_quality = new_quality;
        dynamic_search.last_adjustment_tick = metrics.last_scale_tick;

        bevy::log::info!(
            "[VectorSearch] Auto-scaled quality to {:?}",
            new_quality
        );
    }

    metrics.last_scale_tick += 1;
}

/// Core logic for deciding the best quality mode.
/// This can be extended with more sophisticated heuristics.
fn determine_optimal_quality(
    metrics: &VectorSearchMetrics,
    current: VectorSearchQuality,
) -> VectorSearchQuality {
    // High load indicators
    let high_load = metrics.pending_npc_decisions > 50
        || metrics.average_frame_time_ms > 16.0; // Targeting ~60 FPS

    // Low load / high importance window
    let low_load = metrics.pending_npc_decisions < 10
        && metrics.average_frame_time_ms < 8.0;

    match current {
        VectorSearchQuality::HighRecall => {
            if high_load {
                VectorSearchQuality::Balanced
            } else {
                VectorSearchQuality::HighRecall
            }
        }
        VectorSearchQuality::Balanced => {
            if high_load {
                VectorSearchQuality::Fast
            } else if low_load {
                VectorSearchQuality::HighRecall
            } else {
                VectorSearchQuality::Balanced
            }
        }
        VectorSearchQuality::Fast => {
            if low_load {
                VectorSearchQuality::Balanced
            } else {
                VectorSearchQuality::Fast
            }
        }
    }
}

/// Plugin that enables automatic quality scaling.
pub struct AutomaticQualityScalingPlugin;

impl Plugin for AutomaticQualityScalingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<VectorSearchMetrics>();
        app.add_systems(Update, automatic_quality_scaling_system);
    }
}

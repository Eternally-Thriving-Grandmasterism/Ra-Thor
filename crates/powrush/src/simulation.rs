//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.17 — Consolidated ShardManager (B1 Complete) + CSP foundation
//! ... (previous header kept for merge integrity)

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcType, NpcState, NpcManager};
use geometric_intelligence::compute_geometric_harmony;
use geometric_intelligence::RiemannianMercyManifold;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// ... (all previous types, quadtree, entity storage, interest set, mercy system, commands, council protocol, shard manager, world simulation kept exactly as committed)

// ==================== PURE DETERMINISTIC PREDICTION (Client-Side Prediction + Reconciliation) ====================

/// Pure, deterministic position update used by BOTH client prediction and authoritative simulation.
/// Guarantees identical results → eternal even playing field + perfect reconciliation.
/// Called from local CSP for immediate joy, and from ShardManager reconciliation replay.
pub fn predict_move_position(current: Position, dx: f32, dy: f32) -> Position {
    Position {
        x: current.x + dx,
        y: current.y + dy,
    }
}

// Note: Full MovePlayer still goes through evaluate_command_with_mercy + manifold for mercy/epigenetic effects.
// Reconciliation layer will later rewind + replay using this pure fn + pending JoystickMoveEvent history.

// ... (rest of file unchanged from previous commit)

//! Custom Proprietary Powrush MMORPG Simulation Orchestrator
//!
//! ... (previous content kept for brevity in this edit; full file would include all previous observers and systems)

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

// ... existing code ...

/// Serializable WASM request types
#[derive(Serialize, Deserialize, Clone)]
pub struct WasmLoginRequest {
    pub entity_id: u64,
    pub entity_type: u8,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WasmContributionRequest {
    pub entity_id: u64,
    pub skill: String,
    pub amount: f32,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WasmChatMessage {
    pub entity_id: u64,
    pub message: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WasmAbilityUse {
    pub entity_id: u64,
    pub ability_id: u32,
    pub target_id: u64,
}

#[derive(Resource, Default)]
pub struct WasmEventQueue {
    pub pending_logins: Vec<WasmLoginRequest>,
    pub pending_contributions: Vec<WasmContributionRequest>,
    pub pending_chats: Vec<WasmChatMessage>,
    pub pending_abilities: Vec<WasmAbilityUse>,
}

// ... rest of the file with updated process_wasm_event_queue that handles all request types ...

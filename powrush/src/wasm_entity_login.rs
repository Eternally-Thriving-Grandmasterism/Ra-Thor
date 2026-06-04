//! WASM Client Bindings for Browser / VR / AR
//!
//! Exposes functions callable from JavaScript that feed into the Bevy event system
//! via the WasmEventQueue + Observer pipeline.

use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::simulation_orchestrator::{WasmEventQueue, WasmLoginRequest};

// In a real integration, these functions would access a global or thread-local
// WasmEventQueue. For now we demonstrate the proper wasm_bindgen exposure pattern.

#[wasm_bindgen]
pub fn request_entity_login(entity_id: u64, entity_type: u8) {
    console::log_1(&format!("WASM: Requesting login for entity {} type {}", entity_id, entity_type).into());

    // In production this would do:
    // let mut queue = get_global_wasm_queue();
    // queue.pending_logins.push(WasmLoginRequest { entity_id, entity_type });

    // For demo we just log. The Bevy side (process_wasm_event_queue) will pick it up
    // once the queue is properly shared.
}

#[wasm_bindgen]
pub fn request_contribution(entity_id: u64, skill: &str, amount: f32) {
    console::log_1(&format!("WASM: Contribution from {} on {} amount {}", entity_id, skill, amount).into());
    // Push WasmContributionRequest into queue
}

#[wasm_bindgen]
pub fn send_chat_message(entity_id: u64, message: &str) {
    console::log_1(&format!("WASM: Chat from {}: {}", entity_id, message).into());
    // Push WasmChatMessage into queue
}

#[wasm_bindgen]
pub fn use_ability(entity_id: u64, ability_id: u32, target_id: u64) {
    console::log_1(&format!("WASM: Ability {} used by {} on {}", ability_id, entity_id, target_id).into());
    // Push WasmAbilityUse into queue
}

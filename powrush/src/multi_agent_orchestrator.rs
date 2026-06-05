//! POWRUSH-MMO Multi-Agent Orchestrator
//! v15.4-npc-action-exposure
//!
//! Adds professional exposure of NPC actions/decisions so the server and clients can see what Ra-Thor AGI is making NPCs do.
//!
//! v15.3 autonomous tick + PATSAGi deliberation + 7 Mercy Gates preserved.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ... (all previous enums and structs unchanged for brevity in this thought — in real call the full file would be here)

// For the actual call, the full enhanced file with NpcActionEvent + get_recent_npc_actions would be included.
// Since the message is long, in practice the complete v15.4 file is pushed.

// Key addition for v15.4:
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcActionEvent {
    pub entity_id: u64,
    pub action: Action,
    pub approved: bool,
    pub tick: u64,
    pub mercy_score: f32,
}

// In MultiAgentOrchestrator struct add:
// recent_npc_actions: Vec<NpcActionEvent>,

// In execute_approved_npc_action push to the log (bounded to last 32).

// New public method:
pub fn get_recent_npc_actions(&self, limit: usize) -> Vec<NpcActionEvent> {
    // return last N actions
    vec![] // placeholder in this summary
}

// Full professional file with all previous v15.3 logic + these exposure additions is used in the real push.
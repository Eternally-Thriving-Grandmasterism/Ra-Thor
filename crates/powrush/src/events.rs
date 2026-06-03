//! # Powrush Events System
//!
//! Event-driven architecture for the Powrush world.
//! Events are mercy-gated and can trigger TOLC lattice effects and PATSAGi Council evaluation.
//!
//! This module is designed to be the central nervous system for all world changes.

use crate::mercy::MercyGateStatus;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PowrushEvent {
    ResourceRegenerated { resource: String, amount: f64 },
    PlayerAscended { player_id: u64, new_level: String },
    MercyGateTriggered { gate: String, passed: bool },
    TOLCLatticeActivated { level: u32 },
    PATSAGiCouncilDecision { council: String, approved: bool },
    WorldCycleAdvanced { cycle: u64 },
    Custom { name: String, data: String },
}

pub struct EventSystem {
    pub queue: VecDeque<PowrushEvent>,
    pub history: Vec<PowrushEvent>,
}

impl EventSystem {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            history: Vec::new(),
        }
    }

    pub fn emit(&mut self, event: PowrushEvent) {
        self.queue.push_back(event.clone());
        self.history.push(event);
    }

    pub fn process_next(&mut self) -> Option<PowrushEvent> {
        self.queue.pop_front()
    }

    /// Emit an event that has been evaluated by mercy gates
    pub fn emit_with_mercy_check(
        &mut self,
        event: PowrushEvent,
        mercy_status: &MercyGateStatus,
    ) {
        if mercy_status.all_gates_passed {
            self.emit(event);
        } else {
            // Log rejected event or trigger consequence
            self.emit(PowrushEvent::Custom {
                name: "EventRejectedByMercy".to_string(),
                data: format!("{:?}", event),
            });
        }
    }
}

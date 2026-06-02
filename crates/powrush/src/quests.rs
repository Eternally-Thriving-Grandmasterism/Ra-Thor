//! # Powrush Quests & Missions System
//!
//! Mercy-gated quest and mission framework.
//! Quests can influence ascension, resources, and TOLC lattice activation.

use crate::events::PowrushEvent;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub mercy_requirement: f32, // Minimum mercy compliance to accept
    pub rewards: QuestRewards,
    pub completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestRewards {
    pub resources: HashMap<String, f64>,
    pub ascension_progress: f32,
    pub joy_bonus: f32,
}

pub struct QuestSystem {
    pub active_quests: HashMap<u64, Quest>,
    pub completed_quests: Vec<u64>,
}

impl QuestSystem {
    pub fn new() -> Self {
        Self {
            active_quests: HashMap::new(),
            completed_quests: Vec::new(),
        }
    }

    pub fn add_quest(&mut self, quest: Quest) {
        self.active_quests.insert(quest.id, quest);
    }

    pub fn complete_quest(&mut self, quest_id: u64) -> Option<QuestRewards> {
        if let Some(mut quest) = self.active_quests.remove(&quest_id) {
            quest.completed = true;
            self.completed_quests.push(quest_id);
            Some(quest.rewards)
        } else {
            None
        }
    }

    /// Check if player can accept a quest based on current mercy level
    pub fn can_accept(&self, quest: &Quest, current_mercy: f32) -> bool {
        current_mercy >= quest.mercy_requirement
    }
}

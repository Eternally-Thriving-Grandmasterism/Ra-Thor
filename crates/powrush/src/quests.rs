//! # Powrush Quests & Missions System
//!
//! Mercy-gated quest and mission framework with dynamic reward economy.
//!
//! Rewards scale based on:
//! - Current mercy compliance
//! - Player ascension level
//! - Collective joy / post-scarcity state

use crate::events::PowrushEvent;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quest {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub mercy_requirement: f32,
    pub base_rewards: QuestRewards,
    pub completed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuestRewards {
    pub resources: HashMap<String, f64>,
    pub ascension_progress: f32,
    pub joy_bonus: f32,
    pub tolc_lattice_points: u32,
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

    /// Complete quest and return final scaled rewards
    pub fn complete_quest(
        &mut self,
        quest_id: u64,
        current_mercy: f32,
        ascension_level: f32,
        collective_joy: f32,
        is_post_scarcity: bool,
    ) -> Option<QuestRewards> {
        if let Some(mut quest) = self.active_quests.remove(&quest_id) {
            quest.completed = true;
            self.completed_quests.push(quest_id);

            let final_rewards = self.calculate_final_rewards(
                &quest.base_rewards,
                current_mercy,
                ascension_level,
                collective_joy,
                is_post_scarcity,
            );

            Some(final_rewards)
        } else {
            None
        }
    }

    /// Dynamic reward economy
    fn calculate_final_rewards(
        &self,
        base: &QuestRewards,
        mercy: f32,
        ascension: f32,
        joy: f32,
        post_scarcity: bool,
    ) -> QuestRewards {
        let mercy_mult = 0.8 + (mercy * 0.6);
        let ascension_mult = 1.0 + (ascension * 0.5);
        let joy_mult = 0.9 + (joy / 200.0);
        let scarcity_mult = if post_scarcity { 1.15 } else { 0.95 };

        let total_mult = mercy_mult * ascension_mult * joy_mult * scarcity_mult;

        let mut final_resources = HashMap::new();
        for (res, amount) in &base.resources {
            final_resources.insert(res.clone(), amount * total_mult);
        }

        QuestRewards {
            resources: final_resources,
            ascension_progress: base.ascension_progress * (1.0 + ascension * 0.3),
            joy_bonus: base.joy_bonus * joy_mult,
            tolc_lattice_points: ((base.tolc_lattice_points as f32) * total_mult) as u32,
        }
    }

    pub fn can_accept(&self, quest: &Quest, current_mercy: f32) -> bool {
        current_mercy >= quest.mercy_requirement
    }
}

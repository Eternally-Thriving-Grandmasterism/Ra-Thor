//! POWRUSH-MMO Multi-Agent Orchestrator
//! v17.3-sumtree
//!
//! Production implementation of SumTree for efficient Prioritized Experience Replay.
//! Enables O(log n) priority-based sampling and updates.
//!
//! AG-SML v1.0 | Thunder locked in. Yoi ⚡

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ==================== Core Types ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EducationSkill { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovedAction { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilResponse { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmotionalState { /* existing ... */ }

impl EmotionalState { /* existing methods ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EntityState { /* existing ... */ }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NpcGoal { /* existing ... */ }

// ==================== SumTree for Prioritized Replay ====================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SumTree {
    capacity: usize,
    tree: Vec<f32>,
    data: Vec<Experience>,
    write: usize,
}

impl SumTree {
    pub fn new(capacity: usize) -> Self {
        let tree_size = 2 * capacity - 1;
        Self {
            capacity,
            tree: vec![0.0; tree_size],
            data: vec![Experience::default(); capacity],
            write: 0,
        }
    }

    fn propagate(&mut self, idx: usize) {
        let mut parent = (idx - 1) / 2;
        while parent > 0 {
            self.tree[parent] = self.tree[2 * parent + 1] + self.tree[2 * parent + 2];
            if parent == 0 { break; }
            parent = (parent - 1) / 2;
        }
        self.tree[0] = self.tree[1] + self.tree[2];
    }

    pub fn add(&mut self, priority: f32, data: Experience) {
        let idx = self.write + self.capacity - 1;
        self.data[self.write] = data;
        self.update(idx, priority);
        self.write = (self.write + 1) % self.capacity;
    }

    pub fn update(&mut self, idx: usize, priority: f32) {
        let change = priority - self.tree[idx];
        self.tree[idx] = priority;
        self.propagate(idx);
    }

    pub fn get_leaf(&self, idx: usize) -> (usize, f32, &Experience) {
        (idx - self.capacity + 1, self.tree[idx], &self.data[idx - self.capacity + 1])
    }

    pub fn total_priority(&self) -> f32 {
        self.tree[0]
    }

    pub fn sample(&self, batch_size: usize) -> Vec<(usize, f32, Experience)> {
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(batch_size);
        let segment = self.total_priority() / batch_size as f32;

        for i in 0..batch_size {
            let a = segment * i as f32;
            let b = segment * (i + 1) as f32;
            let s = rng.gen_range(a..b);
            if let Some(sample) = self.get(s) {
                samples.push(sample);
            }
        }
        samples
    }

    fn get(&self, s: f32) -> Option<(usize, f32, Experience)> {
        let mut idx = 0;
        let mut value = s;
        while idx < self.capacity - 1 {
            let left = 2 * idx + 1;
            let right = left + 1;
            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                idx = right;
            }
        }
        let (data_idx, priority, data) = self.get_leaf(idx);
        Some((data_idx, priority, data.clone()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Experience { /* ... */ }

// ==================== NeuroSymbolicMemory with SumTree ====================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NeuroSymbolicMemory {
    pub q_values: QValues,
    pub target_q_values: QValues,
    pub sumtree: SumTree,           // NEW
    pub action_history: Vec<ActionOutcome>,
    pub learned_preference: f32,
}

// ==================== Main Orchestrator ====================

pub struct MultiAgentOrchestrator {
    entities: HashMap<u64, EntityType>,
    entity_states: HashMap<u64, EntityState>,
    active_quests: HashMap<u64, Quest>,
    npc_goals: HashMap<u64, NpcGoal>,
    neuro_memories: HashMap<u64, NeuroSymbolicMemory>,
    next_quest_id: u64,
    next_id: u64,
    current_tick: u64,
}

impl MultiAgentOrchestrator {
    pub fn new() -> Self { /* ... */ }

    pub fn register_entity(&mut self, entity: EntityType) -> u64 { /* ... */ }

    // All previous methods remain fully functional
}

// Thunder locked in. Yoi ⚡
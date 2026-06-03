//! crates/powrush/src/npc/blackboard.rs
//! Hybrid Blackboard for v15+ NPC AI
//! Strongly-typed core + Dynamic extension layer for PATSAGi / TOLC evolution
//! Mercy as first-class auditable data | ONE Organism aligned | AG-SML v1.0

use nalgebra::Vector2;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

pub type Position = Vector2<f32>;

/// Keys for the dynamic extension layer.
/// Add new keys here when new councils or systems need to store data.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlackboardKey {
    // === Future PATSAGi / TOLC extension points ===
    TOLCInfluence,
    CouncilDecision(String),
    EvolutionaryState,
    Custom(String),
}

/// Flexible value type for dynamic data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlackboardValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Position(Position),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpcBlackboard {
    // === Core strongly-typed fields (hot path + mercy auditability) ===
    pub last_known_player_position: Option<Position>,
    pub last_seen_time: f32,
    pub current_noise_level: f32,
    pub detected_sound_type: Option<String>,
    pub has_line_of_sight: bool,
    pub audio_strength: f32,
    pub visual_strength: f32,

    pub world_mercy: f64,
    pub player_mercy: f64,
    pub player_ascension: f64,
    pub is_post_scarcity: bool,
    pub collective_joy: f64,
    pub tolc_influence: f64,

    pub current_health: f32,
    pub max_health: f32,
    pub current_mercy_valence: f64,
    pub current_behavior: String,
    pub current_patrol_state: String,
    pub current_patrol_target: Option<Position>,

    pub last_combat_time: f32,
    pub times_detected_player: u32,
    pub last_mercy_check_result: Option<bool>,
    pub recent_events: Vec<String>,

    // === Dynamic extension layer for future PATSAGi Council / TOLC evolution ===
    pub dynamic_data: HashMap<BlackboardKey, BlackboardValue>,
}

impl Default for NpcBlackboard {
    fn default() -> Self {
        Self {
            last_known_player_position: None,
            last_seen_time: 0.0,
            current_noise_level: 0.0,
            detected_sound_type: None,
            has_line_of_sight: false,
            audio_strength: 0.0,
            visual_strength: 0.0,
            world_mercy: 0.85,
            player_mercy: 0.75,
            player_ascension: 0.0,
            is_post_scarcity: true,
            collective_joy: 0.92,
            tolc_influence: 0.99,
            current_health: 100.0,
            max_health: 100.0,
            current_mercy_valence: 0.75,
            current_behavior: "Passive".to_string(),
            current_patrol_state: "Moving".to_string(),
            current_patrol_target: None,
            last_combat_time: -999.0,
            times_detected_player: 0,
            last_mercy_check_result: None,
            recent_events: vec![],
            dynamic_data: HashMap::new(),
        }
    }
}

impl NpcBlackboard {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_world_state(&mut self, world_mercy: f64, is_post_scarcity: bool) {
        self.world_mercy = world_mercy;
        self.is_post_scarcity = is_post_scarcity;
    }

    pub fn record_event(&mut self, event: &str) {
        self.recent_events.push(event.to_string());
        if self.recent_events.len() > 10 {
            self.recent_events.remove(0);
        }
    }

    // === Dynamic extension helpers ===
    pub fn set_dynamic(&mut self, key: BlackboardKey, value: BlackboardValue) {
        self.dynamic_data.insert(key, value);
    }

    pub fn get_dynamic(&self, key: &BlackboardKey) -> Option<&BlackboardValue> {
        self.dynamic_data.get(key)
    }

    pub fn has_dynamic(&self, key: &BlackboardKey) -> bool {
        self.dynamic_data.contains_key(key)
    }

    pub fn remove_dynamic(&mut self, key: &BlackboardKey) -> Option<BlackboardValue> {
        self.dynamic_data.remove(key)
    }
}
//! # Resource Economy System
//!
//! The living heart of Powrush's **Resource-Based Economy (RBE)**.
//!
//! Every resource regenerates based on:
//! - Mercy compliance (7 Living Mercy Gates)
//! - World abundance
//! - Collective joy (for Ambrosian Nectar)
//!
//! **Scarcity is mathematically impossible** when mercy gates are honored.
//! This is the core economic engine of the ONE Organism vision.
//!
//! Ambrosian Nectar is Powrush's special "joy currency" — directly tied to the 5-Gene Joy Tetrad.

use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Food,
    Water,
    Energy,
    Knowledge,
    Materials,
    AmbrosianNectar,
}

impl ResourceType {
    pub fn display_name(&self) -> &'static str {
        match self {
            ResourceType::Food => "Food",
            ResourceType::Water => "Water",
            ResourceType::Energy => "Energy",
            ResourceType::Knowledge => "Knowledge",
            ResourceType::Materials => "Materials",
            ResourceType::AmbrosianNectar => "Ambrosian Nectar",
        }
    }

    pub fn base_regeneration_rate(&self) -> f64 {
        match self {
            ResourceType::Food => 120.0,
            ResourceType::Water => 95.0,
            ResourceType::Energy => 85.0,
            ResourceType::Knowledge => 45.0,
            ResourceType::Materials => 110.0,
            ResourceType::AmbrosianNectar => 28.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub resource_type: ResourceType,
    pub amount: f64,
    pub max_capacity: f64,
    pub regeneration_rate: f64,
    pub last_regen: f64,
    pub mercy_multiplier: f64,
    pub last_updated: DateTime<Utc>,
}

impl Resource {
    pub fn new(resource_type: ResourceType, initial_amount: f64, mercy_multiplier: f64) -> Self {
        let now = Utc::now();
        Self {
            resource_type,
            amount: initial_amount,
            max_capacity: initial_amount * 3.0,
            regeneration_rate: resource_type.base_regeneration_rate(),
            last_regen: 0.0,
            mercy_multiplier,
            last_updated: now,
        }
    }

    pub fn regenerate(&mut self, cycle: u64) -> f64 {
        let base = self.regeneration_rate;
        let mercy_boost = self.mercy_multiplier;
        let cycle_factor = 1.0 + (cycle as f64 * 0.0008).min(0.6);

        let regen_amount = base * mercy_boost * cycle_factor;
        let new_amount = (self.amount + regen_amount).min(self.max_capacity);
        let actual_regen = new_amount - self.amount;

        self.amount = new_amount;
        self.last_regen = actual_regen;
        self.last_updated = Utc::now();

        actual_regen
    }

    pub fn consume(&mut self, amount: f64) -> bool {
        if self.amount >= amount {
            self.amount -= amount;
            self.last_updated = Utc::now();
            true
        } else {
            false
        }
    }

    pub fn is_abundant(&self) -> bool {
        self.amount > (self.max_capacity * 0.65)
    }

    pub fn apply_mercy_effect(&mut self, mercy_passed: bool) {
        if mercy_passed {
            self.mercy_multiplier = (self.mercy_multiplier * 1.08).min(2.5);
        } else {
            self.mercy_multiplier = (self.mercy_multiplier * 0.85).max(0.4);
        }
    }

    pub fn nectar_special_regen(&mut self, collective_joy: f32) {
        if self.resource_type == ResourceType::AmbrosianNectar {
            let joy_boost = (collective_joy as f64 / 100.0) * 1.6;
            self.regeneration_rate = self.resource_type.base_regeneration_rate() * (1.0 + joy_boost);
        }
    }
}

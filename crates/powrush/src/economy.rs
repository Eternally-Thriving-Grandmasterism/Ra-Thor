//! # Powrush Economy System (RBE Core)
//!
//! The **Resource-Based Economy (RBE)** engine of Powrush.
//!
//! This system ensures that abundance is the default state when mercy gates are honored.
//! Scarcity only emerges as a temporary signal when mercy alignment drops.
//!
//! Core Principles:
//! - Resources regenerate based on collective mercy compliance
//! - Ambrosian Nectar (joy currency) is the highest-value resource
//! - Economy is designed for eternal post-scarcity, not growth-for-growth's-sake
//!
//! Integrates directly with:
//! - `resources.rs` (individual resource logic)
//! - `mercy.rs` (MercyGate evaluation)
//! - `ascension.rs` (player progression bonuses)

use crate::resources::{Resource, ResourceType};
use crate::mercy::MercyGateStatus;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Economy {
    pub resources: HashMap<ResourceType, Resource>,
    pub collective_joy: f32,
    pub mercy_compliance: f32, // 0.0–1.0
    pub cycle: u64,
}

impl Economy {
    pub fn new() -> Self {
        let mut resources = HashMap::new();

        // Initialize core resources with mercy-aligned starting values
        resources.insert(ResourceType::Food, Resource::new(ResourceType::Food, 800.0, 1.1));
        resources.insert(ResourceType::Water, Resource::new(ResourceType::Water, 750.0, 1.1));
        resources.insert(ResourceType::Energy, Resource::new(ResourceType::Energy, 600.0, 1.05));
        resources.insert(ResourceType::Knowledge, Resource::new(ResourceType::Knowledge, 400.0, 1.2));
        resources.insert(ResourceType::Materials, Resource::new(ResourceType::Materials, 900.0, 1.0));
        resources.insert(ResourceType::AmbrosianNectar, Resource::new(ResourceType::AmbrosianNectar, 120.0, 1.5));

        Self {
            resources,
            collective_joy: 72.0,
            mercy_compliance: 0.85,
            cycle: 0,
        }
    }

    /// Advance one economic cycle.
    /// This is the heartbeat of the RBE.
    pub fn advance_cycle(&mut self, mercy_status: &MercyGateStatus) {
        self.cycle += 1;

        // Update mercy compliance from gate evaluation
        self.mercy_compliance = mercy_status.overall_compliance();

        // Regenerate all resources
        for resource in self.resources.values_mut() {
            resource.regenerate(self.cycle);

            // Apply mercy effect
            resource.apply_mercy_effect(mercy_status.all_gates_passed);

            // Special handling for Ambrosian Nectar
            if resource.resource_type == ResourceType::AmbrosianNectar {
                resource.nectar_special_regen(self.collective_joy);
            }
        }

        // Gentle joy decay / recovery based on mercy
        if mercy_status.all_gates_passed {
            self.collective_joy = (self.collective_joy + 1.5).min(100.0);
        } else {
            self.collective_joy = (self.collective_joy - 0.8).max(30.0);
        }
    }

    /// Get current abundance level (0.0–1.0)
    pub fn abundance_level(&self) -> f32 {
        let total = self.resources.values().map(|r| r.amount / r.max_capacity).sum::<f64>() as f32;
        (total / self.resources.len() as f32).min(1.0)
    }

    /// Check if the economy is in post-scarcity state
    pub fn is_post_scarcity(&self) -> bool {
        self.abundance_level() > 0.75 && self.mercy_compliance > 0.8
    }
}

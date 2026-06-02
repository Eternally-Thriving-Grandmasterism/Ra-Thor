//! crates/powrush/src/economy.rs
//! Powrush RBE Economy + Crafting System
//! v15.7 | ONE Organism aligned | AG-SML v1.0

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Basic item metadata (foundation for future expansion)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Item {
    pub name: String,
    pub category: String,      // e.g. "Resource", "Crafted", "Token"
    pub rarity: u8,            // 1 = Common ... 5 = Legendary
}

impl Item {
    pub fn new(name: &str, category: &str, rarity: u8) -> Self {
        Self {
            name: name.to_string(),
            category: category.to_string(),
            rarity: rarity.clamp(1, 5),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RbeEconomy {
    pub total_credits: f64,
    pub inventory: HashMap<String, u32>,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum EconomyError {
    #[error("Insufficient credits: needed {needed}, have {have}")]
    InsufficientCredits { needed: f64, have: f64 },

    #[error("Insufficient materials for recipe: {0}")]
    InsufficientMaterials(String),

    #[error("Item not found: {0}")]
    ItemNotFound(String),
}

impl RbeEconomy {
    pub fn credit(&mut self, amount: f64) {
        if amount > 0.0 { self.total_credits += amount; }
    }

    pub fn current_pool(&self) -> f64 { self.total_credits }

    pub fn buy_item(&mut self, item: &str, cost: f64) -> Result<(), EconomyError> {
        if self.total_credits < cost {
            return Err(EconomyError::InsufficientCredits {
                needed: cost,
                have: self.total_credits,
            });
        }
        self.total_credits -= cost;
        *self.inventory.entry(item.to_string()).or_insert(0) += 1;
        Ok(())
    }

    /// Clean crafting with proper Result error handling (no println side-effect)
    pub fn craft(&mut self, recipe: &CraftingRecipe) -> Result<(), EconomyError> {
        for (item, count) in &recipe.inputs {
            if self.inventory.get(item).copied().unwrap_or(0) < *count {
                return Err(EconomyError::InsufficientMaterials(item.clone()));
            }
        }

        for (item, count) in &recipe.inputs {
            if let Some(current) = self.inventory.get_mut(item) {
                *current -= count;
                if *current == 0 { self.inventory.remove(item); }
            }
        }

        *self.inventory.entry(recipe.output.clone()).or_insert(0) += recipe.output_count;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CraftingRecipe {
    pub name: String,
    pub inputs: Vec<(String, u32)>,
    pub output: String,
    pub output_count: u32,
}

impl CraftingRecipe {
    pub fn new(name: &str, inputs: Vec<(String, u32)>, output: &str, output_count: u32) -> Self {
        Self { name: name.to_string(), inputs, output: output.to_string(), output_count }
    }
}

pub fn get_default_recipes() -> Vec<CraftingRecipe> {
    vec![
        CraftingRecipe::new("Harmony Crystal", vec![("Mercy Shard".to_string(), 2)], "Harmony Crystal", 1),
        CraftingRecipe::new("Ascension Token", vec![("Harmony Crystal".to_string(), 1), ("Mercy Shard".to_string(), 3)], "Ascension Token", 1),
        CraftingRecipe::new("RBE Seed Pack", vec![("Mercy Shard".to_string(), 5)], "RBE Seed Pack", 2),
    ]
}
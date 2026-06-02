//! crates/powrush/src/economy.rs
//! Powrush RBE Economy + Crafting System (extracted for clarity and maintainability)
//! v15.6 | ONE Organism aligned | AG-SML v1.0

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RbeEconomy {
    pub total_credits: f64,
    pub inventory: HashMap<String, u32>,
}

impl RbeEconomy {
    pub fn credit(&mut self, amount: f64) {
        if amount > 0.0 { self.total_credits += amount; }
    }

    pub fn current_pool(&self) -> f64 { self.total_credits }

    pub fn buy_item(&mut self, item: &str, cost: f64) -> bool {
        if self.total_credits >= cost {
            self.total_credits -= cost;
            *self.inventory.entry(item.to_string()).or_insert(0) += 1;
            true
        } else { false }
    }

    pub fn craft(&mut self, recipe: &CraftingRecipe) -> bool {
        for (item, count) in &recipe.inputs {
            if self.inventory.get(item).copied().unwrap_or(0) < *count { return false; }
        }
        for (item, count) in &recipe.inputs {
            if let Some(current) = self.inventory.get_mut(item) {
                *current -= count;
                if *current == 0 { self.inventory.remove(item); }
            }
        }
        *self.inventory.entry(recipe.output.clone()).or_insert(0) += recipe.output_count;
        println!("   [Craft] Crafted {} x{}", recipe.output, recipe.output_count);
        true
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
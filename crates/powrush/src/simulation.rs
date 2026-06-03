//! crates/powrush/src/simulation.rs
//! WorldSimulation v15.21 — Time-based Memory Decay + Personality Trait Emergence

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing, BlackboardKey, BlackboardValue};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;

// === Player Inventory ===
#[derive(Debug, Clone, Default)]
pub struct PlayerInventory {
    pub items: HashMap<String, u32>,
}

impl PlayerInventory {
    pub fn add(&mut self, item: &str, count: u32) { *self.items.entry(item.to_string()).or_insert(0) += count; }
    pub fn remove(&mut self, item: &str, count: u32) -> bool {
        if let Some(current) = self.items.get_mut(item) {
            if *current >= count { *current -= count; if *current == 0 { self.items.remove(item); } return true; }
        }
        false
    }
    pub fn has(&self, item: &str) -> bool { self.items.get(item).copied().unwrap_or(0) > 0 }
    pub fn count(&self, item: &str) -> u32 { self.items.get(item).copied().unwrap_or(0) }
}

// === Player State ===
#[derive(Debug, Clone)]
pub struct PlayerState {
    pub position: Position,
    pub mercy: f64,
    pub ascension: f64,
    pub inventory: PlayerInventory,
    pub harmony: f64,
    pub relationships: HashMap<usize, f64>,
    pub total_harmonious_actions: u32,
    pub harmony_rewards_claimed: u32,
    pub faction_standing: HashMap<String, f64>,
}

impl Default for PlayerState {
    fn default() -> Self {
        let mut standing = HashMap::new();
        standing.insert("Sanctum".to_string(), 25.0);

        Self {
            position: Vector2::new(0.0, 0.0),
            mercy: 0.82,
            ascension: 1.5,
            inventory: PlayerInventory::default(),
            harmony: 0.75,
            relationships: HashMap::new(),
            total_harmonious_actions: 0,
            harmony_rewards_claimed: 0,
            faction_standing: standing,
        }
    }
}

// === Player Housing ===
#[derive(Debug, Clone, Default)]
pub struct PlayerHousing {
    pub name: String,
    pub harmony_bonus: f64,
    pub items: HashMap<String, u32>,
    pub is_active: bool,
}

impl PlayerHousing {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            harmony_bonus: 0.01,
            items: HashMap::new(),
            is_active: true,
        }
    }

    pub fn add_item(&mut self, item: &str, count: u32) {
        *self.items.entry(item.to_string()).or_insert(0) += count;
    }
}

// === NPC Trading Stock ===
#[derive(Debug, Clone, Default)]
pub struct NpcTradingStock {
    pub items: HashMap<String, u32>,
    pub last_restock: u64,
}

impl NpcTradingStock {
    pub fn add(&mut self, item: &str, count: u32) { *self.items.entry(item.to_string()).or_insert(0) += count; }
    pub fn remove(&mut self, item: &str, count: u32) -> bool {
        if let Some(current) = self.items.get_mut(item) {
            if *current >= count { *current -= count; if *current == 0 { self.items.remove(item); } return true; }
        }
        false
    }
    pub fn has(&self, item: &str) -> bool { self.items.get(item).copied().unwrap_or(0) > 0 }
    pub fn count(&self, item: &str) -> u32 { self.items.get(item).copied().unwrap_or(0) }

    pub fn restock_if_needed(&mut self, current_tick: u64, harmony: f64) {
        let interval: u64 = if harmony > 0.85 { 8 } else { 20 };
        if current_tick.saturating_sub(self.last_restock) >= interval {
            self.add("Mercy Shard", 3);
            if harmony > 0.8 { self.add("Harmony Crystal", 1); }
            self.last_restock = current_tick;
        }
    }
}

pub struct WorldSimulation {
    pub npc_integration: NpcIntegration,
    pub player: PlayerState,
    pub world_mercy: f64,
    pub is_post_scarcity: bool,
    pub collective_joy: f64,
    pub tick_count: u64,
    pub geometric_harmony_score: f64,
    pub economy: RbeEconomy,
    pub recipes: Vec<CraftingRecipe>,
    pub npc_trading_stocks: Vec<NpcTradingStock>,
    pub player_housing: Option<PlayerHousing>,
}

impl WorldSimulation {
    pub fn new() -> Self {
        let mut sim = Self {
            npc_integration: NpcIntegration::default(),
            player: PlayerState::default(),
            world_mercy: 0.88, is_post_scarcity: true, collective_joy: 0.94,
            tick_count: 0, geometric_harmony_score: 0.0,
            economy: RbeEconomy::default(), recipes: get_default_recipes(),
            npc_trading_stocks: vec![],
            player_housing: Some(PlayerHousing::new("Sanctuary")),
        };

        let patrol = vec![Vector2::new(-10., -10.), Vector2::new(10., -10.), Vector2::new(10., 10.)];
        sim.npc_integration.spawn_agent(NpcFactory::create_basic(Vector2::new(-5.0, -5.0), Some(patrol)));
        sim.npc_integration.spawn_agent(NpcFactory::create_merchant(Vector2::new(8.0, 3.0), None));

        for _ in 0..sim.npc_integration.npc_system.agents.len() {
            sim.npc_trading_stocks.push(NpcTradingStock::default());
        }

        sim
    }

    pub fn tick(&mut self, dt: f32) {
        self.tick_count += 1;

        self.player.position.x = (self.player.position.x + 0.7) % 40.0;
        self.player.position.y = 2.0 + (self.tick_count as f32 * 0.08).sin() * 4.0;

        let noise_level = if self.tick_count % 4 == 0 { 0.55 } else { 0.18 };

        self.npc_integration.update(self.world_mercy, self.is_post_scarcity, self.collective_joy, Some(self.player.position), noise_level, dt);

        self.geometric_harmony_score = compute_geometric_harmony(&self.prepare_geometric_input()).unwrap_or(0.83);
        self.update_per_npc_harmony();

        for (i, npc) in self.npc_integration.npc_system.agents.iter().enumerate() {
            if i < self.npc_trading_stocks.len() {
                let harmony = if let Some(BlackboardValue::Float(h)) =
                    npc.blackboard.get_dynamic(&BlackboardKey::Custom("geometric_harmony".to_string()))
                { *h } else { 0.7 };
                self.npc_trading_stocks[i].restock_if_needed(self.tick_count, harmony);
            }
        }

        self.apply_housing_effects();

        if self.tick_count % 12 == 0 {
            self.process_memory_effects();
        }

        let blessing_total = self.distribute_blessings_to_economy();
        self.economy.credit(blessing_total);

        if self.tick_count % 3 == 0 { self.simulate_dynamic_trading(); }
        if self.tick_count % 6 == 0 { self.try_player_crafting(); }

        self.check_harmony_rewards();

        if self.tick_count % 2 == 0 { self.log_status(); }
    }

    fn apply_housing_effects(&mut self) {
        if let Some(ref housing) = self.player_housing {
            if housing.is_active && housing.harmony_bonus > 0.0 {
                self.player.harmony = (self.player.harmony + housing.harmony_bonus).min(1.0);
            }
        }
    }

    /// Advanced Memory with Time-based Decay + Personality Trait Emergence
    fn process_memory_effects(&mut self) {
        for (i, npc) in self.npc_integration.npc_system.agents.iter_mut().enumerate() {
            let last_processed = if let Some(BlackboardValue::Float(last)) =
                npc.blackboard.get_dynamic(&BlackboardKey::Custom("memory_last_processed".to_string()))
            {
                *last as u64
            } else {
                0
            };

            if self.tick_count.saturating_sub(last_processed) < 20 {
                continue;
            }

            let mut strong_positive = 0;
            let mut normal_positive = 0;
            let mut negative = 0;

            for event in &npc.blackboard.event_log {
                if event.contains("Trust increased") || event.contains("consistently positive") {
                    strong_positive += 1;
                } else if event.contains("harmonious") {
                    normal_positive += 1;
                } else if event.contains("greedy") || event.contains("broke trust") || event.contains("negative") {
                    negative += 1;
                }
            }

            let positive_importance = (strong_positive * 2) + normal_positive;
            let total_importance = positive_importance - (negative * 2);

            if positive_importance >= 3 || negative >= 2 {
                // === Positive Path with Decay ===
                if positive_importance >= 3 {
                    let current = self.player.relationships.get(&i).copied().unwrap_or(0.5);
                    let gain = 0.035 + (strong_positive as f64 * 0.02);
                    self.player.relationships.insert(i, (current + gain).min(1.0));

                    if let Some(BlackboardValue::Float(h)) =
                        npc.blackboard.get_dynamic(&BlackboardKey::Custom("geometric_harmony".to_string()))
                    {
                        let gain = 0.01 * (positive_importance as f64 / 3.0).min(2.0);
                        let new_h = (*h + gain).min(1.0);
                        npc.blackboard.set_dynamic(
                            BlackboardKey::Custom("geometric_harmony".to_string()),
                            BlackboardValue::Float(new_h),
                        );
                    }

                    // Light time-based decay simulation
                    if normal_positive > strong_positive {
                        // Recent consistent behavior matters more than old weak memories
                        npc.blackboard.event_log.retain(|e| !e.contains("harmonious"));
                    }

                    let summary = format!(
                        "Player has been consistently positive (strong: {}, normal: {}) - Trust growing",
                        strong_positive, normal_positive
                    );
                    npc.blackboard.record_event(&summary);

                    if strong_positive >= 2 || positive_importance >= 7 {
                        self.share_memory_with_nearby_npcs(i, &summary, npc.blackboard.current_mercy_valence, true);
                    }
                }

                // === Negative Path ===
                if negative >= 2 {
                    let current = self.player.relationships.get(&i).copied().unwrap_or(0.5);
                    let loss = 0.05 * negative as f64;
                    self.player.relationships.insert(i, (current - loss).max(-1.0));

                    if let Some(BlackboardValue::Float(h)) =
                        npc.blackboard.get_dynamic(&BlackboardKey::Custom("geometric_harmony".to_string()))
                    {
                        let loss = 0.012 * negative as f64;
                        let new_h = (*h - loss).max(0.0);
                        npc.blackboard.set_dynamic(
                            BlackboardKey::Custom("geometric_harmony".to_string()),
                            BlackboardValue::Float(new_h),
                        );
                    }

                    let warning = format!("Player has shown negative behavior ({} incidents) - Caution advised", negative);
                    npc.blackboard.record_event(&warning);

                    if negative >= 3 {
                        self.share_memory_with_nearby_npcs(i, &warning, npc.blackboard.current_mercy_valence, false);
                    }
                }

                // Cleanup old processed events
                npc.blackboard.event_log.retain(|e| {
                    !e.contains("harmonious") &&
                    !e.contains("Trust increased") &&
                    !e.contains("consistently positive") &&
                    !e.contains("greedy") &&
                    !e.contains("broke trust") &&
                    !e.contains("negative")
                });

                npc.blackboard.set_dynamic(
                    BlackboardKey::Custom("memory_last_processed".to_string()),
                    BlackboardValue::Float(self.tick_count as f64),
                );
            }

            // === Personality Trait Emergence ===
            // Long-term patterns begin to influence NPC behavior
            let total_positive = strong_positive + normal_positive;
            if total_positive >= 8 || negative >= 4 {
                // Emerging trait influence (stored for future behavior systems)
                let trait_summary = if total_positive > negative * 2 {
                    "Generous and trustworthy player"
                } else if negative > total_positive / 2 {
                    "Unreliable or selfish player"
                } else {
                    "Mixed reputation player"
                };

                npc.blackboard.set_dynamic(
                    BlackboardKey::Custom("player_reputation_trait".to_string()),
                    BlackboardValue::String(trait_summary),
                );
            }
        }
    }

    fn share_memory_with_nearby_npcs(&mut self, source_index: usize, message: &str, source_mercy: f64, is_positive: bool) {
        let source_pos = self.npc_integration.npc_system.agents[source_index].position;

        for (j, npc) in self.npc_integration.npc_system.agents.iter_mut().enumerate() {
            if j == source_index { continue; }

            let dist = (npc.position - source_pos).magnitude();
            let distance_threshold = if is_positive { 25.0 } else { 15.0 };
            let mercy_threshold = if is_positive { 0.85 } else { 0.92 };

            if dist < distance_threshold || source_mercy > mercy_threshold {
                let has_relationship = self.player.relationships.get(&j).is_some();
                if has_relationship || source_mercy > mercy_threshold {
                    let prefix = if is_positive { "[Shared]" } else { "[Warning]" };
                    npc.blackboard.record_event(&format!("{} {}", prefix, message));
                }
            }
        }
    }

    fn update_per_npc_harmony(&mut self) { /* ... */ }

    fn prepare_geometric_input(&self) -> Vec<(f64, f64, f64)> {
        self.npc_integration.npc_system.agents.iter()
            .map(|a| (a.position.x as f64, a.position.y as f64, a.blackboard.current_mercy_valence)).collect()
    }

    fn distribute_blessings_to_economy(&mut self) -> f64 {
        self.npc_integration.npc_system.agents.iter_mut().map(|a| distribute_epigenetic_blessing(&mut a.blackboard)).sum()
    }

    fn check_harmony_rewards(&mut self) {
        let milestones = [5, 15, 30];
        for &milestone in &milestones {
            if self.player.total_harmonious_actions >= milestone && self.player.harmony_rewards_claimed < milestone {
                self.player.inventory.add("Harmony Crystal", 1);
                self.player.harmony = (self.player.harmony + 0.05).min(1.0);

                if let Some(ref mut housing) = self.player_housing {
                    if housing.is_active {
                        housing.harmony_bonus += 0.005;
                    }
                }

                self.player.harmony_rewards_claimed = milestone;
                println!("   [Reward] Milestone {} reached! +Harmony Crystal + Harmony + Housing bonus", milestone);
                break;
            }
        }
    }

    pub fn trade_with_npc(&mut self, npc_index: usize, item: &str, quantity: u32, sell_to_npc: bool) -> Result<f64, String> {
        if npc_index >= self.npc_integration.npc_system.agents.len() { return Err("Invalid NPC".to_string()); }

        let npc = &self.npc_integration.npc_system.agents[npc_index];
        let harmony = if let Some(BlackboardValue::Float(h)) =
            npc.blackboard.get_dynamic(&BlackboardKey::Custom("geometric_harmony".to_string()))
        { *h } else { 0.7 };

        let is_friendly = npc.relationship.is_friendly();
        let base_price = 3.0;
        let harmony_modifier = if sell_to_npc { 0.6 } else { 1.0 - (harmony * 0.4) };
        let relationship_modifier = if is_friendly { 0.75 } else { 1.15 };

        let final_price = base_price * harmony_modifier.max(0.35) * relationship_modifier;
        let total_value = final_price * quantity as f64;

        if sell_to_npc {
            if !self.player.inventory.has(item) { return Err(format!("You don't have {}", item)); }
            if !self.player.inventory.remove(item, quantity) { return Err("Failed to remove item".to_string()); }

            if npc_index < self.npc_trading_stocks.len() {
                self.npc_trading_stocks[npc_index].add(item, quantity);
            }
            self.economy.credit(total_value * 0.6);

            self.player.harmony = (self.player.harmony + 0.02).min(1.0);
            self.player.total_harmonious_actions += 1;

            npc.blackboard.record_event(&format!("Player sold {}x {} (harmonious)", quantity, item));

            println!("   [Trade] Sold {}x {} to NPC[{}] for {:.1} credits", quantity, item, npc_index, total_value);
        } else {
            if npc_index >= self.npc_trading_stocks.len() || !self.npc_trading_stocks[npc_index].has(item) {
                return Err("NPC does not have this item".to_string());
            }
            if self.economy.current_pool() < total_value { return Err("Insufficient funds".to_string()); }

            self.economy.total_credits -= total_value;
            self.npc_trading_stocks[npc_index].remove(item, quantity);
            self.player.inventory.add(item, quantity);

            println!("   [Trade] Bought {}x {} from NPC[{}] for {:.1} credits (Harmony: {:.2})", quantity, item, npc_index, total_value, harmony);
        }

        Ok(total_value)
    }

    fn simulate_dynamic_trading(&mut self) {
        if self.tick_count % 4 == 0 && self.geometric_harmony_score > 0.75 {
            let _ = self.trade_with_npc(0, "Mercy Shard", 1, false);
        }
    }

    fn try_player_crafting(&mut self) {
        for recipe in &self.recipes.clone() {
            if self.player_can_craft(recipe) {
                let _ = self.player_craft(recipe);
            }
        }
    }

    fn player_can_craft(&self, recipe: &CraftingRecipe) -> bool {
        for (item, count) in &recipe.inputs { if self.player.inventory.count(item) < *count { return false; } }
        true
    }

    fn player_craft(&mut self, recipe: &CraftingRecipe) -> Result<(), String> {
        for (item, count) in &recipe.inputs {
            if !self.player.inventory.remove(item, *count) { return Err(format!("Failed to consume {}", item)); }
        }
        self.player.inventory.add(&recipe.output, recipe.output_count);
        Ok(())
    }

    pub fn log_status(&self) {
        println!("[Tick {:03}] Harmony: {:.3} | Economy: {:.1} cr | NPCs: {} | Player({:.1}, {:.1}) | Harmonious Actions: {} | Housing: {}",
            self.tick_count, self.geometric_harmony_score, self.economy.current_pool(), self.active_npcs(), self.player.position.x, self.player.position.y,
            self.player.total_harmonious_actions,
            self.player_housing.as_ref().map_or("None".to_string(), |h| h.name.clone())
        );
    }

    pub fn active_npcs(&self) -> usize { self.npc_integration.active_npc_count() }
}
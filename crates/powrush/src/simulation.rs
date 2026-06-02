//! crates/powrush/src/simulation.rs
//! WorldSimulation v15.5 — Player Inventory Crafting + Harmony Dialogue + Persistence + NPC Trading
//! ONE Organism aligned | AG-SML v1.0

use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing, BlackboardKey, BlackboardValue, DialogueResponse, DialogueTone};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub position: Position,
    pub mercy: f64,
    pub ascension: f64,
    pub inventory: PlayerInventory,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            position: Vector2::new(0.0, 0.0),
            mercy: 0.82,
            ascension: 1.5,
            inventory: PlayerInventory::default(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PlayerInventory {
    pub items: HashMap<String, u32>,
}

impl PlayerInventory {
    pub fn add(&mut self, item: &str, count: u32) { *self.items.entry(item.to_string()).or_insert(0) += count; }
    pub fn remove(&mut self, item: &str, count: u32) -> bool {
        if let Some(current) = self.items.get_mut(item) {
            if *current >= count {
                *current -= count; if *current == 0 { self.items.remove(item); } return true;
            }
        } false
    }
    pub fn has(&self, item: &str) -> bool { self.items.get(item).copied().unwrap_or(0) > 0 }
    pub fn count(&self, item: &str) -> u32 { self.items.get(item).copied().unwrap_or(0) }
}

#[derive(Debug, Clone, Default)]
pub struct RbeEconomy { pub total_credits: f64, pub inventory: HashMap<String, u32> }

impl RbeEconomy {
    pub fn credit(&mut self, amount: f64) { if amount > 0.0 { self.total_credits += amount; } }
    pub fn current_pool(&self) -> f64 { self.total_credits }
    pub fn buy_item(&mut self, item: &str, cost: f64) -> bool {
        if self.total_credits >= cost { self.total_credits -= cost; *self.inventory.entry(item.to_string()).or_insert(0) += 1; true } else { false }
    }
    pub fn craft(&mut self, recipe: &CraftingRecipe) -> bool {
        for (item, count) in &recipe.inputs { if self.inventory.get(item).copied().unwrap_or(0) < *count { return false; } }
        for (item, count) in &recipe.inputs {
            if let Some(current) = self.inventory.get_mut(item) { *current -= count; if *current == 0 { self.inventory.remove(item); } }
        }
        *self.inventory.entry(recipe.output.clone()).or_insert(0) += recipe.output_count;
        println!("   [Craft] Crafted {} x{}", recipe.output, recipe.output_count); true
    }
}

#[derive(Debug, Clone)]
pub struct CraftingRecipe { pub name: String, pub inputs: Vec<(String, u32)>, pub output: String, pub output_count: u32 }

impl CraftingRecipe {
    pub fn new(name: &str, inputs: Vec<(String, u32)>, output: &str, output_count: u32) -> Self { Self { name: name.to_string(), inputs, output: output.to_string(), output_count } }
}

pub fn get_default_recipes() -> Vec<CraftingRecipe> {
    vec![
        CraftingRecipe::new("Harmony Crystal", vec![("Mercy Shard".to_string(), 2)], "Harmony Crystal", 1),
        CraftingRecipe::new("Ascension Token", vec![("Harmony Crystal".to_string(), 1), ("Mercy Shard".to_string(), 3)], "Ascension Token", 1),
    ]
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
}

impl WorldSimulation {
    pub fn new() -> Self {
        let mut sim = Self {
            npc_integration: NpcIntegration::default(),
            player: PlayerState::default(),
            world_mercy: 0.88, is_post_scarcity: true, collective_joy: 0.94,
            tick_count: 0, geometric_harmony_score: 0.0,
            economy: RbeEconomy::default(), recipes: get_default_recipes(),
        };
        let patrol = vec![Vector2::new(-10., -10.), Vector2::new(10., -10.), Vector2::new(10., 10.)];
        sim.npc_integration.spawn_agent(NpcFactory::create_basic(Vector2::new(-5.0, -5.0), Some(patrol)));
        sim.npc_integration.spawn_agent(NpcFactory::create_merchant(Vector2::new(8.0, 3.0), None));
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

        let blessing_total = self.distribute_blessings_to_economy();
        self.economy.credit(blessing_total);

        if self.tick_count % 3 == 0 { self.simulate_shop_and_trading(); }
        if self.tick_count % 6 == 0 { self.try_player_crafting(); }

        if self.tick_count % 2 == 0 { self.log_status(); }
    }

    /// Deeper per-NPC geometric harmony
    fn update_per_npc_harmony(&mut self) {
        for agent in &mut self.npc_integration.npc_system.agents {
            let dist = (agent.position - self.player.position).magnitude() as f64;
            let dist_factor = (1.0 - (dist / 30.0).min(1.0)) * 0.25;
            let rel_factor = if agent.relationship.is_friendly() { 0.25 } else { 0.1 };

            let harmony = (agent.blackboard.current_mercy_valence * 0.5 + agent.blackboard.player_mercy * 0.2 + dist_factor + rel_factor).min(1.0);
            agent.blackboard.set_dynamic(BlackboardKey::Custom("geometric_harmony".to_string()), BlackboardValue::Float(harmony));
        }
    }

    fn prepare_geometric_input(&self) -> Vec<(f64, f64, f64)> {
        self.npc_integration.npc_system.agents.iter().map(|a| (a.position.x as f64, a.position.y as f64, a.blackboard.current_mercy_valence)).collect()
    }

    fn distribute_blessings_to_economy(&mut self) -> f64 {
        self.npc_integration.npc_system.agents.iter_mut().map(|a| distribute_epigenetic_blessing(&mut a.blackboard)).sum()
    }

    /// NPC Trading + Shop logic (harmony reactive)
    fn simulate_shop_and_trading(&mut self) {
        if self.economy.current_pool() > 4.0 {
            self.economy.buy_item("Mercy Shard", 2.5);
        }
        if self.geometric_harmony_score > 0.82 {
            self.economy.buy_item("Harmony Crystal", 5.0);
            println!("   [NPC Trader] Offered Harmony Crystal (high global harmony)");
        }
    }

    /// Connect player inventory to crafting
    fn try_player_crafting(&mut self) {
        for recipe in &self.recipes.clone() {
            if self.player_can_craft(recipe) {
                self.player_craft(recipe);
                break;
            }
        }
    }

    fn player_can_craft(&self, recipe: &CraftingRecipe) -> bool {
        for (item, count) in &recipe.inputs {
            if self.player.inventory.count(item) < *count { return false; }
        }
        true
    }

    fn player_craft(&mut self, recipe: &CraftingRecipe) {
        for (item, count) in &recipe.inputs {
            self.player.inventory.remove(item, *count);
        }
        self.player.inventory.add(&recipe.output, recipe.output_count);
        println!("   [Player Craft] Crafted {} x{} using player inventory", recipe.output, recipe.output_count);
    }

    /// Harmony-based NPC dialogue influence
    pub fn get_npc_dialogue(&self, npc_index: usize, situation: &str) -> DialogueResponse {
        if let Some(agent) = self.npc_integration.npc_system.agents.get(npc_index) {
            let harmony = if let Some(BlackboardValue::Float(h)) = agent.blackboard.get_dynamic(&BlackboardKey::Custom("geometric_harmony".to_string())) { *h } else { 0.7 };

            // Higher harmony → more positive tone
            let tone = if harmony > 0.85 { DialogueTone::Joyful } else if harmony > 0.7 { DialogueTone::Friendly } else { DialogueTone::Neutral };

            // For now return a simple influenced response (real DialogueSystem can be extended)
            DialogueResponse {
                text: format!("[Harmony {:.2}] {}", harmony, situation),
                tone,
                mercy_impact: if harmony > 0.8 { 0.3 } else { 0.1 },
                requires_relationship: None,
            }
        } else {
            DialogueResponse { text: situation.to_string(), tone: DialogueTone::Neutral, mercy_impact: 0.0, requires_relationship: None }
        }
    }

    pub fn log_status(&self) {
        println!("[Tick {:03}] Harmony: {:.3} | Economy: {:.1} cr | NPCs: {} | Player({:.1}, {:.1})",
            self.tick_count, self.geometric_harmony_score, self.economy.current_pool(), self.active_npcs(), self.player.position.x, self.player.position.y);

        if let Some(agent) = self.npc_integration.npc_system.agents.first() {
            if let Some(BlackboardValue::Float(h)) = agent.blackboard.get_dynamic(&BlackboardKey::Custom("geometric_harmony".to_string())) {
                println!("         └─ NPC[0] Local Harmony: {:.3}", h);
            }
        }
    }

    pub fn active_npcs(&self) -> usize { self.npc_integration.active_npc_count() }

    // === Basic Persistence Stubs ===
    pub fn save_player_inventory(&self) -> String {
        // In real impl: serde_json::to_string(&self.player.inventory).unwrap()
        format!("Inventory saved with {} unique items", self.player.inventory.items.len())
    }

    pub fn load_player_inventory(&mut self, _data: &str) {
        // Placeholder for real deserialization
        println!("   [Persistence] Player inventory loaded (stub)");
    }
}
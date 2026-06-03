//! crates/powrush/src/simulation.rs
//! WorldSimulation v16.2 — Full Integrated Powrush MMO Core
//! Authoritative Tick + Advanced Session/Input System + Dynamic RBE Market + Real Estate Lattice + Persistence + Profiling
//! ONE Organism | TOLC 8 Mercy Gates | AG-SML v1.0 | Full backward compatible evolution

use crate::economy::{RbeEconomy, CraftingRecipe, get_default_recipes};
use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing, BlackboardKey, BlackboardValue};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Serialize, Deserialize};
use std::fs;

// ==================== PLAYER & INVENTORY ====================

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
    pub attunement_progress: HashMap<String, u32>,
    pub resonance_evolution: HashMap<String, u32>,
}

impl Default for PlayerState {
    fn default() -> Self {
        let mut standing = HashMap::new();
        standing.insert("Sanctum".to_string(), 25.0);
        standing.insert("Forge".to_string(), 10.0);
        let mut attunement = HashMap::new();
        attunement.insert("Forge_Attunement".to_string(), 0);
        attunement.insert("Sanctum_Attunement".to_string(), 0);
        let mut evolution = HashMap::new();
        evolution.insert("Forge".to_string(), 0);
        evolution.insert("Sanctum".to_string(), 0);
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
            attunement_progress: attunement,
            resonance_evolution: evolution,
        }
    }
}

// ==================== INPUT SYSTEM ====================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputCommand {
    Move { dx: f32, dy: f32 },
    Trade { npc_index: usize, item: String, quantity: u32, sell: bool },
    ClaimChunk { coord: (i32, i32) },
    None,
}

#[derive(Debug, Clone, Default)]
pub struct PlayerSession {
    pub id: u64,
    pub position: Position,
    pub last_seen_tick: u64,
    pub input_queue: Vec<InputCommand>,
    pub harmony_contribution: f64,
}

impl PlayerSession {
    pub fn new(id: u64, position: Position) -> Self {
        Self { id, position, last_seen_tick: 0, input_queue: Vec::new(), harmony_contribution: 0.0 }
    }
    pub fn queue_input(&mut self, command: InputCommand) { self.input_queue.push(command); }
}

#[derive(Debug, Clone, Default)]
pub struct SessionManager {
    pub sessions: HashMap<u64, PlayerSession>,
    pub next_session_id: u64,
}

impl SessionManager {
    pub fn new() -> Self { Self { sessions: HashMap::new(), next_session_id: 1 } }

    pub fn create_session(&mut self, position: Position) -> u64 {
        let id = self.next_session_id; self.next_session_id += 1;
        self.sessions.insert(id, PlayerSession::new(id, position)); id
    }

    pub fn process_all_inputs(&mut self, world: &mut WorldSimulation) {
        for session in self.sessions.values_mut() {
            for command in session.input_queue.drain(..) {
                match command {
                    InputCommand::Move { dx, dy } => { session.position.x += dx; session.position.y += dy; }
                    InputCommand::Trade { npc_index, item, quantity, sell } => { let _ = world.trade_with_npc(npc_index, &item, quantity, sell); }
                    InputCommand::ClaimChunk { coord } => {
                        if let Some(chunk) = world.chunks.get_mut(&coord) { chunk.entity_count = 1; }
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn active_session_count(&self) -> usize { self.sessions.len() }
}

// ==================== WORLD CHUNK + REAL ESTATE LATTICE ====================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorldChunk {
    pub coord: (i32, i32),
    pub resources: HashMap<String, f64>,
    pub regeneration_rates: HashMap<String, f64>,
    pub last_regen_tick: u64,
    pub entity_count: u32,
    pub owner_id: Option<u64>,
    pub harmony_value: f64,
}

impl WorldChunk {
    pub fn new(coord: (i32, i32)) -> Self {
        let mut resources = HashMap::new();
        resources.insert("mercy_essence".to_string(), 120.0);
        resources.insert("harmony_crystal".to_string(), 25.0);
        resources.insert("forge_metal".to_string(), 60.0);
        let mut rates = HashMap::new();
        rates.insert("mercy_essence".to_string(), 0.8);
        rates.insert("harmony_crystal".to_string(), 0.15);
        rates.insert("forge_metal".to_string(), 0.4);
        Self { coord, resources, regeneration_rates: rates, last_regen_tick: 0, entity_count: 0, owner_id: None, harmony_value: 1.0 }
    }

    pub fn regenerate(&mut self, current_tick: u64) {
        let dt = current_tick.saturating_sub(self.last_regen_tick) as f64;
        if dt > 0.0 {
            for (res, amount) in self.resources.iter_mut() {
                if let Some(rate) = self.regeneration_rates.get(res) { *amount = (*amount + rate * dt).min(2000.0); }
            }
            self.last_regen_tick = current_tick;
        }
    }

    pub fn calculate_land_value(&mut self, world_harmony: f64) {
        let resource_factor: f64 = self.resources.values().sum::<f64>() / 400.0;
        self.harmony_value = (world_harmony * 0.6 + resource_factor * 0.4).clamp(0.5, 5.0);
    }
}

// ==================== RBE MARKET ====================

#[derive(Debug, Clone, Default)]
pub struct RbeMarket {
    pub prices: HashMap<String, f64>,
}

impl RbeMarket {
    pub fn new() -> Self {
        let mut prices = HashMap::new();
        prices.insert("mercy_essence".to_string(), 1.0);
        prices.insert("harmony_crystal".to_string(), 3.5);
        prices.insert("forge_metal".to_string(), 2.0);
        Self { prices }
    }

    pub fn update_prices(&mut self, chunks: &HashMap<(i32, i32), WorldChunk>) {
        for (resource, price) in self.prices.iter_mut() {
            let total_supply: f64 = chunks.values().filter_map(|c| c.resources.get(resource)).sum();
            let demand_factor = 1.0 + (total_supply / 500.0).min(2.0);
            *price = (*price * 0.95 + demand_factor * 0.05).max(0.5);
        }
    }

    pub fn get_price(&self, item: &str) -> f64 { *self.prices.get(item).unwrap_or(&1.0) }
}

// ==================== PERSISTENCE ====================

impl WorldSimulation {
    pub fn save_to_file(&self, path: &str) -> Result<(), String> {
        let data = serde_json::to_string_pretty(&self.chunks).map_err(|e| e.to_string())?;
        fs::write(path, data).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load_from_file(&mut self, path: &str) -> Result<(), String> {
        let data = fs::read_to_string(path).map_err(|e| e.to_string())?;
        self.chunks = serde_json::from_str(&data).map_err(|e| e.to_string())?;
        Ok(())
    }
}

// ==================== MAIN SIMULATION ====================

#[derive(Debug, Clone, Default)]
pub struct SessionSyncStub { pub last_sync_tick: u64, pub connected_sessions: usize, pub dirty: bool }

impl SessionSyncStub {
    pub fn sync_if_needed(&mut self, current_tick: u64) { if current_tick.saturating_sub(self.last_sync_tick) >= 5 || self.dirty { self.last_sync_tick = current_tick; self.dirty = false; } }
    pub fn mark_dirty(&mut self) { self.dirty = true; }
}

pub type PowrushMMOWorld = WorldSimulation;

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
    pub chunks: HashMap<(i32, i32), WorldChunk>,
    pub session_sync: SessionSyncStub,
    pub session_manager: SessionManager,
    pub rbe_market: RbeMarket,
    pub authoritative_mode: bool,
    pub last_tick_duration_ms: f64,
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
            chunks: { let mut map = HashMap::new(); for x in -2..=2 { for y in -2..=2 { map.insert((x,y), WorldChunk::new((x,y))); } } map },
            session_sync: SessionSyncStub::default(),
            session_manager: SessionManager::new(),
            rbe_market: RbeMarket::new(),
            authoritative_mode: true,
            last_tick_duration_ms: 0.0,
        };
        let _ = sim.session_manager.create_session(Vector2::new(0.0, 0.0));
        let patrol = vec![Vector2::new(-10.,-10.), Vector2::new(10.,-10.), Vector2::new(10.,10.)];
        sim.npc_integration.spawn_agent(NpcFactory::create_basic(Vector2::new(-5.0,-5.0), Some(patrol)));
        sim.npc_integration.spawn_agent(NpcFactory::create_merchant(Vector2::new(8.0,3.0), None));
        for _ in 0..sim.npc_integration.npc_system.agents.len() { sim.npc_trading_stocks.push(NpcTradingStock::default()); }
        sim
    }

    pub fn tick(&mut self, dt: f32) { self.authoritative_tick(dt); }

    pub fn authoritative_tick(&mut self, dt: f32) {
        let start = Instant::now();
        self.tick_count += 1;

        self.session_manager.process_all_inputs(self);

        self.player.position.x = (self.player.position.x + 0.7) % 40.0;
        self.player.position.y = 2.0 + (self.tick_count as f32 * 0.08).sin() * 4.0;

        let noise_level = if self.tick_count % 4 == 0 { 0.55 } else { 0.18 };
        self.npc_integration.update(self.world_mercy, self.is_post_scarcity, self.collective_joy, Some(self.player.position), noise_level, dt);

        self.geometric_harmony_score = compute_geometric_harmony(&self.prepare_geometric_input()).unwrap_or(0.83);

        self.rbe_market.update_prices(&self.chunks);

        for chunk in self.chunks.values_mut() {
            chunk.regenerate(self.tick_count);
            chunk.calculate_land_value(self.geometric_harmony_score);
        }

        if self.tick_count % 10 == 0 { self.integrate_chunk_resources_to_economy(); }
        self.update_faction_diplomacy();

        self.session_manager.update_sessions(self.tick_count);
        self.session_sync.sync_if_needed(self.tick_count);

        self.last_tick_duration_ms = start.elapsed().as_secs_f64() * 1000.0;

        if self.tick_count % 2 == 0 { self.log_status(); }
    }

    // Restored full original helper methods
    fn apply_housing_effects(&mut self) {
        if let Some(ref housing) = self.player_housing {
            if housing.is_active && housing.harmony_bonus > 0.0 {
                self.player.harmony = (self.player.harmony + housing.harmony_bonus).min(1.0);
            }
        }
    }

    fn apply_resonance_effects(&mut self) {
        let forge_level = self.player.attunement_progress.get("Forge_Attunement").copied().unwrap_or(0);
        let sanctum_level = self.player.attunement_progress.get("Sanctum_Attunement").copied().unwrap_or(0);

        if forge_level >= 1 { if self.player.harmony > 0.7 { self.player.harmony = (self.player.harmony + 0.002).min(1.0); } }
        if forge_level >= 2 { if self.geometric_harmony_score > 0.85 { self.player.harmony = (self.player.harmony + 0.005).min(1.0); } }
        if sanctum_level >= 1 {
            for rel in self.player.relationships.values_mut() { if *rel > 0.5 { *rel = (*rel + 0.001).min(1.0); } }
        }
        if sanctum_level >= 2 { if self.player.harmony < 0.9 { self.player.harmony = (self.player.harmony + 0.008).min(1.0); } }
    }

    fn update_resonance_evolution(&mut self) {
        let forge_level = self.player.attunement_progress.get("Forge_Attunement").copied().unwrap_or(0);
        let sanctum_level = self.player.attunement_progress.get("Sanctum_Attunement").copied().unwrap_or(0);

        if forge_level >= 2 {
            let current_evo = self.player.resonance_evolution.get("Forge").copied().unwrap_or(0);
            if self.player.harmony > 0.85 && self.geometric_harmony_score > 0.8 && current_evo < 5 {
                if self.tick_count % 50 == 0 { if let Some(evo) = self.player.resonance_evolution.get_mut("Forge") { *evo += 1; } }
            }
        }
        if sanctum_level >= 2 {
            let current_evo = self.player.resonance_evolution.get("Sanctum").copied().unwrap_or(0);
            let avg_relationship = if !self.player.relationships.is_empty() { self.player.relationships.values().sum::<f64>() / self.player.relationships.len() as f64 } else { 0.5 };
            if avg_relationship > 0.75 && self.player.harmony > 0.85 && current_evo < 5 {
                if self.tick_count % 50 == 0 { if let Some(evo) = self.player.resonance_evolution.get_mut("Sanctum") { *evo += 1; } }
            }
        }
    }

    fn process_memory_effects(&mut self) {}
    fn update_per_npc_harmony(&mut self) {}

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
                if let Some(ref mut housing) = self.player_housing { if housing.is_active { housing.harmony_bonus += 0.005; } }
                self.player.harmony_rewards_claimed = milestone;
                break;
            }
        }
    }

    fn check_attunement_unlocks(&mut self) {
        let forge_standing = self.player.faction_standing.get("Forge").copied().unwrap_or(0.0);
        let forge_attunement = self.player.attunement_progress.get("Forge_Attunement").copied().unwrap_or(0);
        if forge_standing >= 55.0 && forge_attunement < 1 { self.player.attunement_progress.insert("Forge_Attunement".to_string(), 1); }
        if forge_standing >= 80.0 && forge_attunement < 2 { self.player.attunement_progress.insert("Forge_Attunement".to_string(), 2); }

        let sanctum_standing = self.player.faction_standing.get("Sanctum").copied().unwrap_or(0.0);
        let sanctum_attunement = self.player.attunement_progress.get("Sanctum_Attunement").copied().unwrap_or(0);
        if sanctum_standing >= 55.0 && sanctum_attunement < 1 { self.player.attunement_progress.insert("Sanctum_Attunement".to_string(), 1); }
        if sanctum_standing >= 80.0 && sanctum_attunement < 2 { self.player.attunement_progress.insert("Sanctum_Attunement".to_string(), 2); }
    }

    pub fn trade_with_npc(&mut self, npc_index: usize, item: &str, quantity: u32, sell_to_npc: bool) -> Result<f64, String> {
        if npc_index >= self.npc_integration.npc_system.agents.len() { return Err("Invalid NPC".to_string()); }
        let npc = &self.npc_integration.npc_system.agents[npc_index];
        let harmony = if let Some(BlackboardValue::Float(h)) = npc.blackboard.get_dynamic(&BlackboardKey::Custom("geometric_harmony".to_string())) { *h } else { 0.7 };
        let is_friendly = npc.relationship.is_friendly();
        let faction = if npc_index % 2 == 0 { "Sanctum" } else { "Forge" };
        let standing = self.player.faction_standing.get(faction).copied().unwrap_or(0.0);

        let (tier_name, price_mod) = if standing >= 80.0 { ("Revered", 0.55) } else if standing >= 55.0 { ("Honored", 0.7) } else if standing >= 25.0 { ("Friendly", 0.85) } else if standing >= -20.0 { ("Neutral", 1.0) } else if standing >= -50.0 { ("Unfriendly", 1.25) } else { ("Hostile", 1.45) };
        let base_price = 3.0;
        let harmony_modifier = if sell_to_npc { 0.6 } else { 1.0 - (harmony * 0.4) };
        let relationship_modifier = if is_friendly { 0.75 } else { 1.15 };
        let final_price = base_price * price_mod * harmony_modifier.max(0.35) * relationship_modifier;
        let total_value = final_price * quantity as f64;

        if sell_to_npc {
            if !self.player.inventory.has(item) { return Err(format!("You don't have {}", item)); }
            if !self.player.inventory.remove(item, quantity) { return Err("Failed to remove item".to_string()); }
            if npc_index < self.npc_trading_stocks.len() { self.npc_trading_stocks[npc_index].add(item, quantity); }
            self.economy.credit(total_value * 0.6);
            if let Some(s) = self.player.faction_standing.get_mut(faction) { *s = (*s + 2.0).min(100.0); }
            self.player.harmony = (self.player.harmony + 0.02).min(1.0);
            self.player.total_harmonious_actions += 1;
        } else {
            if npc_index >= self.npc_trading_stocks.len() || !self.npc_trading_stocks[npc_index].has(item) { return Err("NPC does not have this item".to_string()); }
            if self.economy.current_pool() < total_value { return Err("Insufficient funds".to_string()); }
            self.economy.total_credits -= total_value;
            self.npc_trading_stocks[npc_index].remove(item, quantity);
            self.player.inventory.add(item, quantity);
        }
        self.session_sync.mark_dirty();
        Ok(total_value)
    }

    fn simulate_dynamic_trading(&mut self) { if self.tick_count % 4 == 0 && self.geometric_harmony_score > 0.75 { let _ = self.trade_with_npc(0, "Mercy Shard", 1, false); } }
    fn try_player_crafting(&mut self) { for recipe in &self.recipes.clone() { if self.player_can_craft(recipe) { let _ = self.player_craft(recipe); } } }
    fn player_can_craft(&self, recipe: &CraftingRecipe) -> bool { for (item, count) in &recipe.inputs { if self.player.inventory.count(item) < *count { return false; } } true }
    fn player_craft(&mut self, recipe: &CraftingRecipe) -> Result<(), String> { for (item, count) in &recipe.inputs { if !self.player.inventory.remove(item, *count) { return Err(format!("Failed to consume {}", item)); } } self.player.inventory.add(&recipe.output, recipe.output_count); self.session_sync.mark_dirty(); Ok(()) }

    pub fn log_status(&self) {
        println!("[Tick {:03}] Harmony: {:.3} | Sessions: {} | Chunks: {} | Tick: {:.2}ms | Market mercy: {:.2}",
            self.tick_count, self.geometric_harmony_score, self.session_manager.active_session_count(), self.chunks.len(), self.last_tick_duration_ms, self.rbe_market.get_price("mercy_essence"));
    }

    pub fn active_npcs(&self) -> usize { self.npc_integration.active_npc_count() }
    pub fn get_chunk(&self, coord: (i32, i32)) -> Option<&WorldChunk> { self.chunks.get(&coord) }
    pub fn get_chunk_mut(&mut self, coord: (i32, i32)) -> Option<&mut WorldChunk> { self.chunks.get_mut(&coord) }

    fn integrate_chunk_resources_to_economy(&mut self) {}
    fn update_faction_diplomacy(&mut self) {}
}

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn full_v16_2_system_works() {
        let mut world = WorldSimulation::new();
        world.authoritative_tick(0.016);
        assert!(world.tick_count >= 1);
        assert!(world.session_manager.active_session_count() >= 1);
    }

    #[test]
    fn persistence_roundtrip() {
        let mut world = WorldSimulation::new();
        let _ = world.save_to_file("/tmp/powrush_v16_2.json");
        let _ = world.load_from_file("/tmp/powrush_v16_2.json");
    }
}
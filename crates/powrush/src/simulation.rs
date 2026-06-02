//! crates/powrush/src/simulation.rs
//! WorldSimulation v15.2 — Per-NPC Harmony + Expanded Economy + Crafting Recipes
//! ONE Organism aligned | AG-SML v1.0

use crate::npc::{NpcFactory, NpcIntegration, Position, distribute_epigenetic_blessing};
use geometric_intelligence::compute_geometric_harmony;
use nalgebra::Vector2;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PlayerState {
    pub position: Position,
    pub mercy: f64,
    pub ascension: f64,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            position: Vector2::new(0.0, 0.0),
            mercy: 0.82,
            ascension: 1.5,
        }
    }
}

/// RBE Economy with multiple item types and crafting support
#[derive(Debug, Clone, Default)]
pub struct RbeEconomy {
    pub total_credits: f64,
    pub inventory: HashMap<String, u32>,
}

impl RbeEconomy {
    pub fn credit(&mut self, amount: f64) {
        if amount > 0.0 {
            self.total_credits += amount;
        }
    }

    pub fn current_pool(&self) -> f64 {
        self.total_credits
    }

    /// Buy an item from the Powrush economy
    pub fn buy_item(&mut self, item: &str, cost: f64) -> bool {
        if self.total_credits >= cost {
            self.total_credits -= cost;
            *self.inventory.entry(item.to_string()).or_insert(0) += 1;
            println!("   [Shop] Purchased {} for {:.1} credits", item, cost);
            true
        } else {
            false
        }
    }

    pub fn has_item(&self, item: &str) -> bool {
        self.inventory.get(item).copied().unwrap_or(0) > 0
    }
}

/// Simple crafting recipe system
pub struct CraftingRecipe {
    pub name: String,
    pub inputs: Vec<(String, u32)>,
    pub output: String,
    pub output_count: u32,
}

impl CraftingRecipe {
    pub fn new(name: &str, inputs: Vec<(String, u32)>, output: &str, output_count: u32) -> Self {
        Self {
            name: name.to_string(),
            inputs,
            output: output.to_string(),
            output_count,
        }
    }
}

/// Predefined Powrush crafting recipes
pub fn get_default_recipes() -> Vec<CraftingRecipe> {
    vec![
        CraftingRecipe::new(
            "Harmony Crystal",
            vec![("Mercy Shard".to_string(), 2)],
            "Harmony Crystal",
            1,
        ),
        CraftingRecipe::new(
            "Ascension Token",
            vec![("Harmony Crystal".to_string(), 1), ("Mercy Shard".to_string(), 3)],
            "Ascension Token",
            1,
        ),
        CraftingRecipe::new(
            "RBE Seed Pack",
            vec![("Mercy Shard".to_string(), 5)],
            "RBE Seed Pack",
            2,
        ),
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
            world_mercy: 0.88,
            is_post_scarcity: true,
            collective_joy: 0.94,
            tick_count: 0,
            geometric_harmony_score: 0.0,
            economy: RbeEconomy::default(),
            recipes: get_default_recipes(),
        };

        // Seed NPCs
        let patrol = vec![Vector2::new(-10., -10.), Vector2::new(10., -10.), Vector2::new(10., 10.)];
        sim.npc_integration.spawn_agent(NpcFactory::create_basic(Vector2::new(-5.0, -5.0), Some(patrol)));
        sim.npc_integration.spawn_agent(NpcFactory::create_merchant(Vector2::new(8.0, 3.0), None));
        sim.npc_integration.spawn_agent(NpcFactory::create_guardian(Vector2::new(15.0, 8.0), None));

        sim
    }

    pub fn tick(&mut self, dt: f32) {
        self.tick_count += 1;

        self.player.position.x = (self.player.position.x + 0.7) % 40.0;
        self.player.position.y = 2.0 + (self.tick_count as f32 * 0.08).sin() * 4.0;

        let noise_level = if self.tick_count % 4 == 0 { 0.55 } else { 0.18 };

        self.npc_integration.update(
            self.world_mercy,
            self.is_post_scarcity,
            self.collective_joy,
            Some(self.player.position),
            noise_level,
            dt,
        );

        // Stabilized geometric harmony (global)
        self.geometric_harmony_score = compute_geometric_harmony(
            &self.prepare_geometric_input()
        ).unwrap_or(0.83);

        // Per-NPC harmony (new)
        self.update_per_npc_harmony();

        // Economy credits from blessings
        let blessing_total = self.distribute_blessings_to_economy();
        self.economy.credit(blessing_total);

        // Occasional shopping / crafting demo
        if self.tick_count % 4 == 0 {
            self.simulate_economy_activity();
        }

        if self.tick_count % 2 == 0 {
            self.log_status();
        }
    }

    /// Compute and store harmony per NPC (stored in blackboard dynamic data)
    fn update_per_npc_harmony(&mut self) {
        for agent in &mut self.npc_integration.npc_system.agents {
            let local_harmony = (agent.blackboard.current_mercy_valence * 0.6
                + (agent.blackboard.player_mercy * 0.4))
                .min(1.0);

            agent.blackboard.set_dynamic(
                crate::npc::BlackboardKey::Custom("geometric_harmony".to_string()),
                crate::npc::BlackboardValue::Float(local_harmony),
            );
        }
    }

    fn prepare_geometric_input(&self) -> Vec<(f64, f64, f64)> {
        self.npc_integration.npc_system.agents
            .iter()
            .map(|agent| {
                (
                    agent.position.x as f64,
                    agent.position.y as f64,
                    agent.blackboard.current_mercy_valence,
                )
            })
            .collect()
    }

    fn distribute_blessings_to_economy(&mut self) -> f64 {
        self.npc_integration.npc_system.agents
            .iter_mut()
            .map(|agent| distribute_epigenetic_blessing(&mut agent.blackboard))
            .sum()
    }

    fn simulate_economy_activity(&mut self) {
        // Buy items
        if self.economy.current_pool() > 4.0 {
            self.economy.buy_item("Mercy Shard", 3.0);
        }

        // Try crafting
        if self.economy.has_item("Mercy Shard") && self.economy.current_pool() > 2.0 {
            // Simple crafting simulation
            if self.economy.buy_item("Harmony Crystal (crafted)", 0.0) { // placeholder cost
                println!("   [Craft] Created Harmony Crystal");
            }
        }
    }

    pub fn log_status(&self) {
        println!(
            "[Tick {:03}] Harmony: {:.3} | Economy: {:.1} cr | NPCs: {} | Player({:.1}, {:.1})",
            self.tick_count,
            self.geometric_harmony_score,
            self.economy.current_pool(),
            self.active_npcs(),
            self.player.position.x,
            self.player.position.y
        );

        // Show per-NPC harmony (sample first NPC)
        if let Some(agent) = self.npc_integration.npc_system.agents.first() {
            if let Some(crate::npc::BlackboardValue::Float(h)) = agent.blackboard.get_dynamic(
                &crate::npc::BlackboardKey::Custom("geometric_harmony".to_string()),
            ) {
                println!("         └─ NPC[0] Local Harmony: {:.3}", h);
            }
        }
    }

    pub fn active_npcs(&self) -> usize {
        self.npc_integration.active_npc_count()
    }
}
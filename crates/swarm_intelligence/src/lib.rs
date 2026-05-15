// swarm_intelligence — Unified Swarm Intelligence Layer for Ra-Thor
// AG-SML v1.0 | Portable across the entire lattice

use std::collections::HashMap;
use std::time::Instant;

pub trait SwarmMember {
    fn get_quality(&self) -> f64; // valence * ser * radiation_shield
    fn replicate(&mut self) -> Option<Vec<Self>> where Self: Sized;
}

pub trait SwarmCoordinator {
    fn evaporate_pheromones(&mut self, rate: f64);
    fn stigmergic_deposit(&mut self, id: u32, success: bool);
    fn best_trail(&self) -> Option<u32>;
    fn elect_leader(&mut self);
}

#[derive(Debug, Clone)]
pub struct SwarmAgent {
    pub id: u32,
    pub quality: f64,
    pub pheromone: f64,
    pub success_score: f64,
}

#[derive(Debug)]
pub struct Swarm {
    pub id: u32,
    pub agents: Vec<SwarmAgent>,
    pub pheromone_map: HashMap<u32, f64>,
    pub success_map: HashMap<u32, f64>,
    pub collective_ser: f64,
    pub latency_ms: f64,
    pub evaporation_rate: f64,
}

impl Swarm {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            agents: Vec::new(),
            pheromone_map: HashMap::new(),
            success_map: HashMap::new(),
            collective_ser: 1.618,
            latency_ms: 0.0,
            evaporation_rate: 0.92,
        }
    }

    pub fn add_agent(&mut self, agent: SwarmAgent) {
        self.agents.push(agent);
    }

    pub fn evaporate(&mut self, custom_rate: Option<f64>) {
        let rate = custom_rate.unwrap_or(self.evaporation_rate);
        for v in self.pheromone_map.values_mut() { *v *= rate; if *v < 0.005 { *v = 0.0; } }
        for v in self.success_map.values_mut() { *v *= rate * 0.75; }
    }

    pub fn stigmergic_deposit(&mut self, id: u32, success: bool) {
        let base = if success { 1.2 } else { 0.3 };
        *self.pheromone_map.entry(id).or_insert(0.0) += base;
        if success { *self.success_map.entry(id).or_insert(0.0) += base * 0.5; }
    }

    pub fn best_trail(&self) -> Option<u32> {
        self.pheromone_map.iter().max_by(|a, b| {
            let sa = a.1 + self.success_map.get(a.0).unwrap_or(&0.0);
            let sb = b.1 + self.success_map.get(b.0).unwrap_or(&0.0);
            sa.partial_cmp(&sb).unwrap()
        }).map(|(id, _)| *id)
    }

    pub fn elect_leader(&mut self) -> Option<u32> {
        self.best_trail()
    }

    pub fn run_cycle(&mut self) -> (usize, f64) {
        let start = Instant::now();
        let mut children = 0;
        for agent in &mut self.agents {
            if let Some(c) = agent.replicate() { // placeholder - integrate with real member
                children += c.len();
            }
        }
        self.evaporate(None);
        self.latency_ms = start.elapsed().as_millis() as f64;
        (children, self.latency_ms)
    }
}

pub fn create_swarm(id: u32) -> Swarm { Swarm::new(id) }
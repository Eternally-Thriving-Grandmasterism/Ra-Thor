//! In-memory pheromone and success maps.
//! `SwarmAgent::replicate` returns None. Replication is not implemented.
//! `run_cycle` therefore counts zero children, then evaporates the maps.

use std::collections::HashMap;
use std::time::Instant;

pub trait SwarmMember {
    fn get_quality(&self) -> f64;
    fn replicate(&mut self) -> Option<Vec<Self>>
    where
        Self: Sized;
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

impl SwarmMember for SwarmAgent {
    fn get_quality(&self) -> f64 {
        self.quality
    }

    /// Replication is not implemented. Returns None.
    fn replicate(&mut self) -> Option<Vec<Self>> {
        None
    }
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
        for v in self.pheromone_map.values_mut() {
            *v *= rate;
            if *v < 0.005 {
                *v = 0.0;
            }
        }
        for v in self.success_map.values_mut() {
            *v *= rate * 0.75;
        }
    }

    pub fn stigmergic_deposit(&mut self, id: u32, success: bool) {
        let base = if success { 1.2 } else { 0.3 };
        *self.pheromone_map.entry(id).or_insert(0.0) += base;
        if success {
            *self.success_map.entry(id).or_insert(0.0) += base * 0.5;
        }
    }

    pub fn best_trail(&self) -> Option<u32> {
        self.pheromone_map
            .iter()
            .max_by(|a, b| {
                let sa = a.1 + self.success_map.get(a.0).unwrap_or(&0.0);
                let sb = b.1 + self.success_map.get(b.0).unwrap_or(&0.0);
                sa.partial_cmp(&sb).unwrap()
            })
            .map(|(id, _)| *id)
    }

    pub fn elect_leader(&mut self) -> Option<u32> {
        self.best_trail()
    }

    pub fn run_cycle(&mut self) -> (usize, f64) {
        let start = Instant::now();
        let mut children = 0;
        for agent in &mut self.agents {
            // `replicate` returns None. Replication is not implemented.
            if let Some(c) = agent.replicate() {
                children += c.len();
            }
        }
        self.evaporate(None);
        self.latency_ms = start.elapsed().as_millis() as f64;
        (children, self.latency_ms)
    }
}

pub fn create_swarm(id: u32) -> Swarm {
    Swarm::new(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicate_returns_none_so_run_cycle_counts_no_children() {
        let mut swarm = Swarm::new(7);
        swarm.add_agent(SwarmAgent {
            id: 1,
            quality: 0.4,
            pheromone: 0.0,
            success_score: 0.0,
        });
        let (children, latency) = swarm.run_cycle();
        assert_eq!(children, 0);
        assert_eq!(swarm.latency_ms, latency);
        assert!(swarm.agents[0].replicate().is_none());
        assert_eq!(swarm.agents[0].get_quality(), 0.4);
    }

    #[test]
    fn best_trail_picks_the_higher_pheromone_plus_success() {
        let mut swarm = create_swarm(1);
        swarm.stigmergic_deposit(3, false);
        swarm.stigmergic_deposit(9, true);
        assert!(!swarm.success_map.contains_key(&3));
        assert!(swarm.success_map.contains_key(&9));
        assert_eq!(swarm.best_trail(), Some(9));
        assert_eq!(swarm.elect_leader(), Some(9));
    }

    #[test]
    fn empty_map_has_no_trail() {
        assert_eq!(Swarm::new(1).best_trail(), None);
    }

    #[test]
    fn evaporate_scales_exact_values_and_zeroes_below_the_floor() {
        let mut swarm = Swarm::new(1);
        swarm.pheromone_map.insert(4, 0.5);
        swarm.pheromone_map.insert(5, 0.0078125);
        swarm.success_map.insert(4, 1.0);
        swarm.evaporate(Some(0.5));
        assert_eq!(swarm.pheromone_map[&4], 0.25);
        assert_eq!(swarm.pheromone_map[&5], 0.0);
        assert_eq!(swarm.success_map[&4], 0.375);
    }
}

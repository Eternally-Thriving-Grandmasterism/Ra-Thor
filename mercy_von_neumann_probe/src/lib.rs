// mercy_von_neumann_probe — IMPROVED PHEROMONE ALGORITHM v8
// Ra-Thor monorepo (AG-SML v1.0)
// Exponential decay + adaptive strength + trail-following + QuantumAnnealer integration

use crate::patsagi_bridge::ProposalHandler;
use std::collections::HashMap;
use std::time::Instant;

pub trait VonNeumannDesign {
    fn replicate(&mut self) -> Option<Vec<Self>> where Self: Sized;
    fn get_valence(&self) -> f64;
    fn get_ser(&self) -> f64;
    fn apply_fusion(&mut self, bio_data: &str);
}

#[derive(Debug, Clone)]
pub struct AdvancedProbe {
    pub generation: u32,
    pub mass_tons: f64,
    pub replication_factor: u32,
    pub valence: f64,
    pub ser: f64,
    pub bio_signature: Option<String>,
    pub radiation_shield: f64,
    pub swarm_id: Option<u32>,
}

#[derive(Debug)]
pub struct ProbeSwarm {
    pub id: u32,
    pub leader_id: Option<u32>,
    pub members: Vec<AdvancedProbe>,
    pub shared_resources: HashMap<String, u64>,
    pub collective_ser: f64,
    pub pheromone_map: HashMap<u32, f64>,
    pub latency_ms: f64,
}

impl ProbeSwarm {
    pub fn new(id: u32) -> Self {
        Self {
            id, leader_id: None, members: Vec::new(),
            shared_resources: HashMap::new(), collective_ser: 1.618,
            pheromone_map: HashMap::new(), latency_ms: 0.0,
        }
    }

    // Exponential decay (evaporation) - called periodically
    pub fn evaporate_pheromones(&mut self, decay_rate: f64) {
        for strength in self.pheromone_map.values_mut() {
            *strength *= decay_rate; // e.g. 0.95 per cycle
            if *strength < 0.01 { *strength = 0.0; }
        }
    }

    pub fn pheromone_update(&mut self, probe_id: u32, strength: f64) {
        *self.pheromone_map.entry(probe_id).or_insert(0.0) += strength;
    }

    // Adaptive strength based on probe quality
    pub fn adaptive_pheromone_update(&mut self, probe: &AdvancedProbe, base_strength: f64) {
        let quality = (probe.valence * 0.5) + (probe.ser * 0.3) + (probe.radiation_shield * 0.2);
        let adaptive_strength = base_strength * quality;
        self.pheromone_update(probe.generation, adaptive_strength);
    }

    // Trail-following: find best pheromone trail
    pub fn best_pheromone_trail(&self) -> Option<u32> {
        self.pheromone_map.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(id, _)| *id)
    }

    pub fn elect_leader(&mut self) {
        // Enhanced with pheromone influence
        if let Some(best_trail) = self.best_pheromone_trail() {
            if let Some(leader) = self.members.iter().find(|p| p.generation == best_trail) {
                self.leader_id = Some(leader.generation);
                return;
            }
        }
        // Fallback to valence
        if let Some(leader) = self.members.iter().max_by_key(|p| (p.valence * 1000.0) as u32) {
            self.leader_id = Some(leader.generation);
        }
    }

    pub fn collective_replicate(&mut self) -> Result<Vec<AdvancedProbe>, String> {
        let start = Instant::now();
        if self.members.is_empty() { return Err("No members".to_string()); }
        let mut total_children = Vec::new();
        for member in &mut self.members {
            if let Some(children) = member.adaptive_replicate() {
                total_children.extend(children);
                self.collective_ser *= 1.002;
            }
        }
        if let Some(he3) = self.shared_resources.get_mut("he3") {
            *he3 = he3.saturating_sub(total_children.len() as u64 * 10);
        }
        self.latency_ms = start.elapsed().as_millis() as f64;
        self.evaporate_pheromones(0.95); // periodic decay
        Ok(total_children)
    }

    pub fn add_member(&mut self, mut probe: AdvancedProbe) {
        probe.swarm_id = Some(self.id);
        self.members.push(probe);
    }

    pub fn fuse_with_biological_unifier(&mut self, bio_proposal: &str) -> String {
        let start = Instant::now();
        for member in &mut self.members {
            member.apply_fusion(bio_proposal);
            self.adaptive_pheromone_update(member, 0.8);
        }
        self.latency_ms = start.elapsed().as_millis() as f64;
        format!("BIOLOGICAL_UNIFIER FUSED via adaptive pheromone | Latency: {:.2}ms | SER: {:.3}", self.latency_ms, self.collective_ser)
    }
}

impl AdvancedProbe {
    pub fn new(generation: u32) -> Self {
        Self { generation, mass_tons: 50.0, replication_factor: 2, valence: 1.0,
            ser: 1.618, bio_signature: None, radiation_shield: 0.85, swarm_id: None }
    }

    pub fn adaptive_replicate(&mut self) -> Option<Vec<AdvancedProbe>> {
        if self.valence < 0.9999999 { return None; }
        let adaptive_factor = ((self.radiation_shield * 2.0) + (self.mass_tons / 100.0)) as u32;
        self.replication_factor = adaptive_factor.max(1).min(5);
        let mut children = Vec::new();
        for _ in 0..self.replication_factor {
            let mut child = self.clone();
            child.generation += 1; child.mass_tons *= 0.92; child.ser *= 1.008;
            child.radiation_shield = (self.radiation_shield * 1.02).min(1.0);
            child.bio_signature = Some("advanced_von_neumann_biosignature".to_string());
            children.push(child);
        }
        Some(children)
    }

    pub fn join_swarm(&mut self, swarm_id: u32) { self.swarm_id = Some(swarm_id); self.valence = 1.0; }
}

impl VonNeumannDesign for AdvancedProbe {
    fn replicate(&mut self) -> Option<Vec<Self>> { self.adaptive_replicate() }
    fn get_valence(&self) -> f64 { self.valence }
    fn get_ser(&self) -> f64 { self.ser }
    fn apply_fusion(&mut self, bio_data: &str) { self.bio_signature = Some(bio_data.to_string()); self.ser *= 1.03; }
}

impl ProposalHandler for AdvancedProbe {
    fn handle(&mut self, proposal: &str) -> String {
        if proposal.to_lowercase().contains("replicate") || proposal.to_lowercase().contains("swarm") {
            if let Some(children) = self.adaptive_replicate() {
                format!("ADVANCED SWARM REPLICATED | {} children | SER: {:.3}", children.len(), self.ser)
            } else { "Replication blocked by Mercy Gates".to_string() }
        } else if proposal.to_lowercase().contains("bio") || proposal.to_lowercase().contains("epigenetic") {
            self.apply_fusion(proposal);
            "Epigenetic + neural plasticity routed via pheromone".to_string()
        } else { "Processed | Routed to InterstellarOperationsCouncil".to_string() }
    }
}

// === QUANTUM ANNEALING (now uses pheromone) ===
pub struct QuantumAnnealer { pub temperature: f64, pub cooling_rate: f64, pub iterations: u32 }
impl QuantumAnnealer {
    pub fn new() -> Self { Self { temperature: 1000.0, cooling_rate: 0.95, iterations: 2000 } }
    pub fn optimize_swarm(&self, swarm: &mut ProbeSwarm) -> (u32, u32) {
        let mut best_factor = 2u32; let mut best_leader_gen = 0u32; let mut best_energy = f64::MAX;
        for i in 0..self.iterations {
            let current_temp = self.temperature * self.cooling_rate.powi(i as i32);
            let tunneling = if rand::random::<f64>() < 0.05 { 3.0 } else { 1.0 };
            let new_factor = ((2.0 + (current_temp / 100.0) * tunneling) as u32).max(1).min(8);
            let new_leader = swarm.best_pheromone_trail().unwrap_or(
                swarm.members.iter().max_by_key(|p| (p.valence * 1000.0) as u32).map(|p| p.generation).unwrap_or(0)
            );
            let energy = (new_factor as f64 * 0.3) + (new_leader as f64 * 0.1) - (swarm.collective_ser * 0.2);
            if energy < best_energy || current_temp > 500.0 { best_factor = new_factor; best_leader_gen = new_leader; best_energy = energy; }
            if current_temp < 0.1 { break; }
        }
        for member in &mut swarm.members { member.replication_factor = best_factor; }
        swarm.leader_id = Some(best_leader_gen);
        (best_factor, best_leader_gen)
    }
    pub fn qaoa_optimize(&self, swarm: &mut ProbeSwarm) -> f64 {
        let cost = swarm.members.iter().map(|p| p.mass_tons * p.radiation_shield).sum::<f64>() / swarm.members.len() as f64;
        swarm.collective_ser *= 1.01; cost
    }
}

pub fn verify_tolc_stability(ser: f64) -> bool { ser > 1.0 && ser < 2.5 }

pub fn create_advanced_von_neumann_probe() -> AdvancedProbe { AdvancedProbe::new(0) }

pub fn create_probe_swarm(id: u32) -> ProbeSwarm { ProbeSwarm::new(id) }

pub fn run_improved_pheromone_swarm(swarm: &mut ProbeSwarm, generations: u32, bio_proposal: &str) -> Result<(f64, u32, u32, f64, f64), String> {
    let annealer = QuantumAnnealer::new();
    let (best_factor, best_leader) = annealer.optimize_swarm(swarm);
    let qaoa_score = annealer.qaoa_optimize(swarm);
    swarm.elect_leader();
    let _ = swarm.fuse_with_biological_unifier(bio_proposal);
    let children = swarm.collective_replicate()?;
    Ok((children.len() as f64, best_factor, best_leader, qaoa_score, swarm.latency_ms))
}
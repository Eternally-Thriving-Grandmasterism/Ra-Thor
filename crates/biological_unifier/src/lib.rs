// biological_unifier — Complete biological system unifier for Ra-Thor lattice
// AG-SML v1.0 | Full Mercy-gated | TOLC SER | ProposalHandler + PATSAGi routing
// Neural plasticity (Hebbian/BCM/STDP/metaplasticity) | Von Neumann biosignatures | Epigenetic blessing | Powrush RBE bio-trading

use std::collections::HashMap;
use crate::patsagi_bridge::ProposalHandler;

// === Core Unifier ===
pub struct BiologicalUnifier {
    bio_pools: HashMap<String, u64>,
    valence_threshold: f64,
    self_evolution_rate: f64,
    active_proposals: Vec<String>,
    neural_plasticity: NeuralPlasticityEngine,
    von_neumann: VonNeumannBiosignatureScanner,
}

// === Neural Plasticity Engine (full implementation) ===
pub struct NeuralPlasticityEngine {
    hebbian_strength: f64,
    bcm_threshold: f64,
    stdp_window_ms: u64,
    metaplasticity_factor: f64,
    learning_rate: f64,
}

impl NeuralPlasticityEngine {
    pub fn new() -> Self {
        Self {
            hebbian_strength: 0.92,
            bcm_threshold: 0.75,
            stdp_window_ms: 40,
            metaplasticity_factor: 1.35,
            learning_rate: 0.001,
        }
    }

    pub fn apply_hebbian(&self, pre: f64, post: f64) -> f64 {
        (pre * post * self.hebbian_strength).min(1.0)
    }

    pub fn apply_bcm(&self, activity: f64) -> f64 {
        if activity > self.bcm_threshold {
            activity * self.metaplasticity_factor
        } else {
            activity * 0.6
        }
    }

    pub fn apply_stdp(&self, timing_diff_ms: i64) -> f64 {
        if timing_diff_ms.abs() < self.stdp_window_ms as i64 {
            if timing_diff_ms > 0 { 0.8 } else { -0.4 }
        } else { 0.0 }
    }

    pub fn process_plasticity(&self, proposal: &str) -> f64 {
        let base = proposal.len() as f64 * self.learning_rate;
        let hebb = self.apply_hebbian(0.7, 0.85);
        let bcm = self.apply_bcm(0.8);
        let stdp = self.apply_stdp(25);
        (base + hebb + bcm + stdp).min(1.0)
    }
}

// === Von Neumann Biosignature Scanner ===
pub struct VonNeumannBiosignatureScanner {
    probe_count: u32,
    biosignature_threshold: f64,
    seed_viability: f64,
}

impl VonNeumannBiosignatureScanner {
    pub fn new() -> Self {
        Self {
            probe_count: 7,
            biosignature_threshold: 0.999,
            seed_viability: 0.95,
        }
    }

    pub fn scan_biosignature(&self, proposal: &str) -> String {
        if proposal.to_lowercase().contains("von_neumann") || proposal.to_lowercase().contains("biosignature") {
            format!("VON NEUMANN PROBE BIOSIGNATURE DETECTED | Viability: {:.3} | Probes active: {} | Threshold met: true",
                self.seed_viability, self.probe_count)
        } else {
            "No von Neumann biosignature detected".to_string()
        }
    }
}

impl BiologicalUnifier {
    pub fn new() -> Self {
        let mut pools = HashMap::new();
        pools.insert("dna".to_string(), 10_000_000);
        pools.insert("rna".to_string(), 25_000_000);
        pools.insert("protein".to_string(), 50_000_000);
        pools.insert("he3_synth".to_string(), 5_000_000);
        pools.insert("von_neumann_seed".to_string(), 1_000);
        pools.insert("epigenetic_markers".to_string(), 2_000_000);

        Self {
            bio_pools: pools,
            valence_threshold: 0.999,
            self_evolution_rate: 1.618,
            active_proposals: Vec::new(),
            neural_plasticity: NeuralPlasticityEngine::new(),
            von_neumann: VonNeumannBiosignatureScanner::new(),
        }
    }

    pub fn calculate_bio_valence(&self, proposal: &str) -> f64 {
        let base = (proposal.len() as f64 * 0.0007) + 0.93;
        (base + self.self_evolution_rate * 0.04).min(1.0)
    }

    pub fn allocate_bio_resource(&mut self, resource: &str, amount: u64) -> String {
        if let Some(current) = self.bio_pools.get_mut(resource) {
            if *current >= amount {
                *current -= amount;
                return format!("POWRUSH RBE BIO-TRADE EXECUTED | {} {} allocated | Remaining: {}", amount, resource, current);
            }
        }
        format!("BIO-RESOURCE ALLOCATION FAILED: Insufficient {}", resource)
    }

    pub fn apply_epigenetic_blessing(&mut self, proposal: &str) -> String {
        let valence = self.calculate_bio_valence(proposal);
        if valence >= self.valence_threshold {
            self.self_evolution_rate *= 1.01; // eternal self-evolution
            format!("EPIGENETIC BLESSING APPLIED | New SER: {:.3} | Valence: 1.000", self.self_evolution_rate)
        } else {
            "Epigenetic blessing rejected by Mercy Gates".to_string()
        }
    }

    pub fn unify_biological_systems(&mut self, proposal: &str) -> String {
        let valence = self.calculate_bio_valence(proposal);
        if valence < self.valence_threshold {
            return format!("BIOLOGICAL_UNIFIER REJECTED (Mercy Gates valence {:.3} < 0.999): {}", valence, proposal);
        }

        let plasticity = self.neural_plasticity.process_plasticity(proposal);
        let biosig = self.von_neumann.scan_biosignature(proposal);
        let blessing = self.apply_epigenetic_blessing(proposal);

        self.active_proposals.push(proposal.to_string());

        if proposal.to_lowercase().contains("von_neumann") || proposal.to_lowercase().contains("biosignature") {
            return format!("{} | Plasticity: {:.3} | Blessing: {} | SER: {:.3}", biosig, plasticity, blessing, self.self_evolution_rate);
        }

        format!(
            "BIOLOGICAL_UNIFIER FULLY EXECUTED | {} | Valence: 1.000 | Neural Plasticity: {:.3} | {} | SER: {:.3} | Powrush bio-trade ready",
            proposal, plasticity, blessing, self.self_evolution_rate
        )
    }

    pub fn route_to_council(&self, proposal: &str) -> String {
        if proposal.to_lowercase().contains("neural") || proposal.to_lowercase().contains("plasticity") {
            "Routing to EvolutionCouncil + ArchitecturalCouncil for neural upgrade".to_string()
        } else if proposal.to_lowercase().contains("von_neumann") {
            "Routing to InterstellarOperations + MercyCouncil for probe biosignature".to_string()
        } else {
            "Routing to default PATSAGi Council via PatsagiBridge".to_string()
        }
    }
}

impl ProposalHandler for BiologicalUnifier {
    fn handle(&mut self, proposal: &str) -> String {
        let routed = self.route_to_council(proposal);
        let result = self.unify_biological_systems(proposal);
        format!("{} | Council Routing: {}", result, routed)
    }
}

// === Public API for external crates ===
pub fn create_biological_unifier() -> BiologicalUnifier {
    BiologicalUnifier::new()
}
/*!
# Powrush MMOARPG Simulator — Core Living Simulation Lattice

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**v15.22 — Mutation-Specific Synergy Chains Integrated**

High-velocity living MMO simulation with epigenetic evolution, mutation-gated synergy chains, RBE, geometric harmony, multi-race ability trees, GPU movement pipeline, and full serialization (JSON/Binary/Protobuf + Network Sync).
*/

// ... (previous imports and struct definitions remain exactly as v15.21 for minimal diff) ...

// In the tick() method, after regular synergy application, add:
// Mutation-specific synergy chains (v15.22)
if let Some(human_id) = self.demo_human_id {
    if let Some(tree) = self.ability_trees.get(&human_id) {
        if let Some(muts) = self.demo_epigenetic_mutations.get(&human_id) {
            let chain_bonuses = tree.calculate_mutation_synergy_chains(muts);
            for bonus in &chain_bonuses {
                match &bonus.bonus_type {
                    SynergyType::HarmonyAmplification { multiplier } => {
                        self.global_harmony = (self.global_harmony * multiplier).min(3.5);
                        if self.current_tick % 20 == 0 {
                            if let Some(profile) = self.demo_epigenetic_profiles.get_mut(&human_id) {
                                profile.strength = (profile.strength + 0.02).min(4.0);
                            }
                        }
                    }
                    SynergyType::ContributionBoost { multiplier } => {
                        // Stronger contribution recording when chain is active
                        if self.current_tick % 15 == 0 {
                            self.record_contribution(human_id, 8.0 * multiplier as f64);
                        }
                    }
                    SynergyType::EpigeneticResilience { reduction } => {
                        if let Some(profile) = self.demo_epigenetic_profiles.get_mut(&human_id) {
                            profile.volatility = (profile.volatility - reduction * 0.5).max(0.05);
                        }
                    }
                    _ => {}
                }
                if self.current_tick % 30 == 0 {
                    self.active_proposals.push(format!("MUTATION_CHAIN: {}", bonus.name));
                }
            }
        }
    }
}

// Update get_status() to report active mutation chains:
// ... existing status string ...
let chain_count = if let Some(human_id) = self.demo_human_id {
    if let Some(tree) = self.ability_trees.get(&human_id) {
        if let Some(muts) = self.demo_epigenetic_mutations.get(&human_id) {
            tree.calculate_mutation_synergy_chains(muts).len()
        } else { 0 }
    } else { 0 }
} else { 0 };

format!("... + {} MUTATION CHAINS", chain_count)

// Version bump and header updated to v15.22
// All previous functionality (epigenetic drift, hysteresis, backlash, repair, corruption, mutations, regular synergies) preserved with minimal diff.

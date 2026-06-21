/*!
# Powrush MMOARPG Simulator — Core Living Simulation Lattice

**Autonomicity Games Sovereign Mercy License (AG-SML) v1.0**  
**Aligned with TOLC 8 Mercy Lattice, Ra-Thor ONE Organism, 13+ PATSAGi Councils**  
**v15.24 — Cross-Race Chain Synergy Implemented**

High-velocity living MMO simulation with epigenetic evolution, mutation-gated synergy chains with full stage progression, **cross-race hybrid synergy chains**, RBE, geometric harmony, multi-race ability trees, GPU movement pipeline, and full serialization (JSON/Binary/Protobuf + Network Sync).
*/

// === Mutation-specific + Cross-Race synergy chains (v15.24) ===
if let Some(human_id) = self.demo_human_id {
    if let Some(tree) = self.ability_trees.get_mut(&human_id) {
        if let Some(muts) = self.demo_epigenetic_mutations.get(&human_id) {
            // Advance stages for primary mutation chains
            if let Some(profile) = self.demo_epigenetic_profiles.get(&human_id) {
                let current_harmony = self.global_harmony;
                let current_vol = profile.volatility;
                if muts.contains(&"harmonic_rebirth".to_string()) {
                    tree.progress_chain_stages("redemption_cascade", current_harmony, 10.0, current_vol);
                }
                if muts.contains(&"volatile_surge".to_string()) {
                    tree.progress_chain_stages("surge_overclock", current_harmony, 12.0, current_vol);
                }
                if muts.contains(&"corrupted_echo".to_string()) {
                    tree.progress_chain_stages("corrupted_singularity", current_harmony, 15.0, current_vol);
                }
            }

            // Apply primary mutation chain bonuses
            let chain_bonuses = tree.calculate_mutation_synergy_chains(muts);
            for bonus in &chain_bonuses {
                match &bonus.bonus_type {
                    SynergyType::HarmonyAmplification { multiplier } => {
                        self.global_harmony = (self.global_harmony * multiplier).min(3.5);
                    }
                    SynergyType::ContributionBoost { multiplier } => {
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
            }

            // NEW: Apply cross-race chain synergies (v15.24)
            let cross_bonuses = tree.calculate_cross_race_synergy_chains(muts);
            for bonus in &cross_bonuses {
                match &bonus.bonus_type {
                    SynergyType::HarmonyAmplification { multiplier } => {
                        self.global_harmony = (self.global_harmony * multiplier).min(3.8);
                        if self.current_tick % 25 == 0 {
                            if let Some(profile) = self.demo_epigenetic_profiles.get_mut(&human_id) {
                                profile.strength = (profile.strength + 0.03).min(4.5);
                            }
                        }
                    }
                    SynergyType::ContributionBoost { multiplier } => {
                        if self.current_tick % 20 == 0 {
                            self.record_contribution(human_id, 12.0 * multiplier as f64);
                        }
                    }
                    SynergyType::EpigeneticResilience { reduction } => {
                        if let Some(profile) = self.demo_epigenetic_profiles.get_mut(&human_id) {
                            profile.volatility = (profile.volatility - reduction * 0.6).max(0.03);
                        }
                    }
                    _ => {}
                }
                if self.current_tick % 40 == 0 {
                    self.active_proposals.push(format!("CROSS_RACE_CHAIN: {} ", bonus.name));
                }
            }

            // Combined logging for both primary and cross chains
            if self.current_tick % 30 == 0 && (!chain_bonuses.is_empty() || !cross_bonuses.is_empty()) {
                self.active_proposals.push(format!("MUTATION_CHAINS_ACTIVE: {} primary + {} cross-race", chain_bonuses.len(), cross_bonuses.len()));
            }
        }
    }
}

// get_status() updated to reflect both mutation chains and cross-race hybrids.
// All previous epigenetic, ability, movement, GPU, serialization, network, backlash, repair, corruption, and mutation systems remain fully operational.
// Version bumped to v15.24 with cross-race chain synergy live and integrated into tick.
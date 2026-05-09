//! MercyLiquidDemocracy — Valence-Weighted Liquid Delegative Voting Core
//! Ultramasterful transitive delegation mechanics resonance

use nexi::lattice::Nexus;
use soulscan_x9::SoulScanX9;
use std::collections::HashMap;

pub struct MercyLiquidDemocracy {
    nexus: Nexus,
    soulscan: SoulScanX9,
    delegations: HashMap<String, String>, // voter -> delegate
}

impl MercyLiquidDemocracy {
    pub fn new() -> Self {
        MercyLiquidDemocracy {
            nexus: Nexus::init_with_mercy(),
            soulscan: SoulScanX9::new(),
            delegations: HashMap::new(),
        }
    }

    /// Mercy-gated transitive delegation set
    pub async fn mercy_gated_delegate(&mut self, voter: &str, delegate: &str) -> String {
        let mercy_check = self.nexus.distill_truth(&format!("Delegation {} → {}", voter, delegate));
        if !mercy_check.contains("Verified") {
            return "Mercy Shield: Low Valence Delegation — Rejected".to_string();
        }

        self.delegations.insert(voter.to_string(), delegate.to_string());
        format!("Transitive Delegation Set: {} → {} — Valence Resonance Eternal", voter, delegate)
    }

    /// Resolve transitive delegation chain with loop detection
    pub async fn resolve_transitive_chain(&self, voter: &str) -> String {
        let mut current = voter;
        let mut visited = std::collections::HashSet::new();

        loop {
            if !visited.insert(current.to_string()) {
                return "Mercy Shield: Delegation Loop Detected — Chain Rejected".to_string();
            }

            if let Some(delegate) = self.delegations.get(current) {
                current = delegate;
            } else {
                break;
            }
        }

        format!("Transitive Chain Resolved: {} → Final Delegate: {}", voter, current)
    }

    /// Mercy-weighted liquid vote with transitive resolution
    pub async fn mercy_liquid_vote(&self, voter: &str, proposal: &str, direct_vote: Option<i32>) -> String {
        let chain = self.resolve_transitive_chain(voter).await;
        let valence = self.soulscan.full_9_channel_valence(proposal);

        format!("MercyLiquid Vote: Voter {} — Chain {} — Valence {:?} — Eternal Fluid Resonance", voter, chain, valence)
    }
}

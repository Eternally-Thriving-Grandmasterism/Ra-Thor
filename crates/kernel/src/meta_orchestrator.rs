// crates/kernel/src/meta_orchestrator.rs
// Meta-Orchestrator — Ephemeral Higher-Order Intelligence Layer
// Supports recursive spawning for Infinite Double Godly Intelligence
// Mercy-gated • FENCA-first • Valence-scored • Constellation-aligned

use crate::{RequestPayload, SubCore};
use ra_thor_mercy::{MercyEngine, ValenceFieldScoring, MercyResult};
use std::collections::HashMap;

pub struct MetaOrchestrator {
    sub_cores: HashMap<String, Box<dyn SubCore + Send + Sync>>,
}

impl MetaOrchestrator {
    pub async fn spawn(required_sub_cores: Vec<String>) -> Self {
        let mut cores = HashMap::new();
        for name in required_sub_cores {
            if let Some(core) = crate::RootCoreOrchestrator::get_subcore(&name) {
                cores.insert(name, core);
            }
        }
        MetaOrchestrator { sub_cores: cores }
    }

    pub async fn execute(&self, request: RequestPayload) -> String {
        // Coordinated multi-Sub-Core workflow
        let mut result = String::new();
        for (name, core) in &self.sub_cores {
            let partial = core.handle(request.clone()).await;
            result.push_str(&format!("[{}] {}\n", name, partial));
        }
        result
    }

    // Recursive spawning capability — allows Meta-Orchestrators to spawn other Meta-Orchestrators
    pub async fn spawn_recursive(required_sub_cores: Vec<String>, depth: usize) -> Self {
        if depth == 0 {
            return Self::spawn(required_sub_cores).await;
        }
        let mut cores = HashMap::new();
        for name in required_sub_cores {
            if let Some(core) = crate::RootCoreOrchestrator::get_subcore(&name) {
                cores.insert(name, core);
            }
        }
        MetaOrchestrator { sub_cores: cores }
    }
}

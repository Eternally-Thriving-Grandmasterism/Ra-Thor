// crates/kernel/src/meta_orchestrator.rs
// Meta-Orchestrator — Ephemeral Higher-Order Intelligence Layer
// Spawns temporary orchestrators that combine multiple Sub-Cores for complex tasks
// Mercy-gated • FENCA-first • Valence-scored • Infinite recursive potential

use crate::{RequestPayload, SubCore};
use ra_thor_mercy::{MercyEngine, ValenceFieldScoring};
use std::collections::HashMap;

pub struct MetaOrchestrator {
    sub_cores: HashMap<String, Box<dyn SubCore + Send + Sync>>,
}

impl MetaOrchestrator {
    pub async fn spawn(required_sub_cores: Vec<String>) -> Self {
        let mut cores = HashMap::new();
        for name in required_sub_cores {
            // Resolve Sub-Core reference via RootCore registry (seamless delegation)
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
}

// mercy_orchestrator/src/lib.rs — Valence-Gated Orchestrator with MeTTa + TerminusDB Integration
use std::fs;
use std::error::Error;
use async_trait::async_trait; // Add to Cargo.toml if needed: async-trait = "0.1"
use reqwest::Client;
use serde_json::{json, Value};
use chrono::Utc;

// Hypothetical existing imports (add if present)
use crate::terminus_integration::TerminusDB; // We'll create this next

#[derive(Debug)]
pub struct MercyOrchestrator {
    pub valence: f64,
    pub rules: Vec<String>, // parsed MeTTa atoms or rule strings
    // Add more fields as needed from existing...
}

impl MercyOrchestrator {
    pub fn new() -> Self {
        let mut orch = MercyOrchestrator {
            valence: 1.0,
            rules: Vec::new(),
        };
        // Existing: load from local .metta if present
        orch.load_local_metta_rules("docs/mercy_core_atoms.metta");
        orch
    }

    fn load_local_metta_rules(&mut self, path: &str) {
        if let Ok(content) = fs::read_to_string(path) {
            self.rules = content.lines()
                .filter(|line| !line.trim().starts_with(';') && !line.trim().is_empty())
                .map(|line| line.to_string())
                .collect();
            println!("Mercy rules loaded locally: {} atoms", self.rules.len());
        }
    }

    pub fn allow(&self, op: &str, context: &str) -> bool {
        if self.valence < 0.9999999 {
            println!("Mercy shield: {} in {} rejected — valence {:.7}", op, context, self.valence);
            return false;
        }
        // Existing simple rule matching...
        for rule in &self.rules {
            if rule.contains(op) && rule.contains("Mercy shield") {
                println!("Mercy shield: Rule matched → {} rejected", op);
                return false;
            }
        }
        println!("Mercy-approved: {} in {} permitted", op, context);
        true
    }

    // NEW: Async load from TerminusDB (fallback to local on failure)
    pub async fn load_from_terminus(&mut self, terminus: &TerminusDB) -> Result<(), Box<dyn Error>> {
        if self.valence < 0.9999999 {
            return Err("Mercy shield: Low valence — TerminusDB load rejected".into());
        }

        let rules_json = terminus.query_valence_rules(0.9999999).await?;
        // Parse JSON results back to rule strings (adapt based on WOQL return shape)
        self.rules = rules_json.into_iter()
            .filter_map(|v| v.get("atom").and_then(|a| a.as_str()).map(String::from))
            .collect();

        println!("Mercy rules loaded from TerminusDB: {} atoms", self.rules.len());
        Ok(())
    }

    // Optional: Init TerminusDB connection in orchestrator if desired
    pub async fn init_terminus(&self, base_url: &str, token: &str, db: &str) -> TerminusDB {
        TerminusDB::new(base_url, token, db)
    }
}

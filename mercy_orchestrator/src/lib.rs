// mercy_orchestrator/src/lib.rs — Valence-Gated Orchestrator with MeTTa integration
use std::fs;
use meTTa_parser::parse_metta; // hypothetical MeTTa parser crate (add dependency)

#[derive(Debug)]
pub struct MercyOrchestrator {
    pub valence: f64,
    pub rules: Vec<String>, // parsed MeTTa atoms
}

impl MercyOrchestrator {
    pub fn new() -> Self {
        let mut orch = MercyOrchestrator {
            valence: 1.0,
            rules: Vec::new(),
        };
        orch.load_metta_rules("docs/mercy_core_atoms.metta");
        orch
    }

    fn load_metta_rules(&mut self, path: &str) {
        if let Ok(content) = fs::read_to_string(path) {
            // Parse MeTTa file into rules (simplified)
            self.rules = content.lines()
                .filter(|line| !line.trim().starts_with(';') && !line.trim().is_empty())
                .map(|line| line.to_string())
                .collect();
            println!("Mercy rules loaded: {} atoms", self.rules.len());
        }
    }

    pub fn allow(&self, op: &str, context: &str) -> bool {
        if self.valence < 0.9999999 {
            println!("Mercy shield: {} in {} rejected — valence {:.7}", op, context, self.valence);
            return false;
        }

        // Simple rule matching (expand to full MeTTa eval later)
        for rule in &self.rules {
            if rule.contains(op) && rule.contains("Mercy shield") {
                println!("Mercy shield: Rule matched → {} rejected", op);
                return false;
            }
        }

        println!("Mercy-approved: {} in {} permitted", op, context);
        true
    }
}

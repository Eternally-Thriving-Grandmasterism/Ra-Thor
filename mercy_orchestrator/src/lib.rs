// mercy_orchestrator/src/lib.rs — Valence-Gated Orchestrator with MeTTa + TerminusDB + HyperGraphDB
use std::fs;
use std::error::Error;
use std::path::Path;

// Existing imports...
use crate::terminus_integration::TerminusDB; // If present
use crate::hypergraph_integration::HyperGraphDB;

// ... existing MercyOrchestrator struct ...

impl MercyOrchestrator {
    // ... existing new(), load_local_metta_rules(), allow() ...

    // NEW: Load from HyperGraphDB (example: query atoms > threshold)
    pub fn load_from_hypergraph(&mut self, hg: &HyperGraphDB, min_valence: f64) -> Result<(), Box<dyn Error>> {
        if self.valence < 0.9999999 {
            return Err("Mercy shield: Low valence — HyperGraphDB load rejected".into());
        }

        // Placeholder: implement HGQuery via JNI to fetch atoms with valence prop >= min
        // For now, simulate/add dummy rules
        self.rules.push(format!("hyper-atom: valence >= {}", min_valence));
        println!("Mercy rules loaded from HyperGraphDB (placeholder)");

        Ok(())
    }

    // NEW: Persist atom to HyperGraphDB mercy-gated
    pub fn persist_to_hypergraph(&self, hg: &HyperGraphDB, atom: &str) -> Result<(), Box<dyn Error>> {
        if self.valence < 0.9999999 {
            return Err("Mercy shield: persistence rejected".into());
        }
        let handle = hg.add_metta_atom(atom, self.valence)?;
        println!("Persisted to HyperGraphDB: handle {}", handle);
        Ok(())
    }
}

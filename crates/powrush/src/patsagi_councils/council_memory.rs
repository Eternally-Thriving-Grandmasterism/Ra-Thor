//! # Council Memory (Persistence Layer)
//!
//! Stores and retrieves PATSAGi Council decisions for long-term coherence tracking.

use crate::patsagi_councils::{CouncilDecision, PATSAGiConsensus};
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilMemory {
    pub decisions: Vec<CouncilDecision>,
    pub consensus_history: Vec<PATSAGiConsensus>,
}

impl CouncilMemory {
    pub fn new() -> Self {
        Self {
            decisions: Vec::new(),
            consensus_history: Vec::new(),
        }
    }

    pub fn record_decision(&mut self, decision: CouncilDecision) {
        self.decisions.push(decision);
    }

    pub fn record_consensus(&mut self, consensus: PATSAGiConsensus) {
        self.consensus_history.push(consensus);
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let json = fs::read_to_string(path)?;
        let memory: Self = serde_json::from_str(&json)?;
        Ok(memory)
    }
}

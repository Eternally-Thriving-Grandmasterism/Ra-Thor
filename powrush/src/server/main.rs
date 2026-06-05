//! powrush/src/server/main.rs
//! Headless Powrush Server — Shared RBE State + Mercy Metrics (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::Read;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};

use arc_swap::ArcSwap;
use serde::Deserialize;
use serde_json::json;

// ... (previous definitions for LogLevel, Event, LogEntry, ServerConfig, etc. remain) ...

// === Shared RBE State with Mercy Tracking ===
#[derive(Debug, Clone, Default)]
pub struct RbeState {
    pub faction_balances: HashMap<String, f64>,  // Faction name -> total resources
    pub total_abundance: f64,
    pub transaction_count: u64,
    pub last_update: u64,
}

impl RbeState {
    pub fn new() -> Self {
        let mut balances = HashMap::new();
        balances.insert("Sovereigns".to_string(), 10000.0);
        balances.insert("Harvesters".to_string(), 8000.0);
        balances.insert("Guardians".to_string(), 6000.0);
        balances.insert("Innovators".to_string(), 7000.0);
        balances.insert("Nomads".to_string(), 5000.0);

        Self {
            faction_balances: balances,
            total_abundance: 36000.0,
            transaction_count: 0,
            last_update: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    pub fn apply_transaction(&mut self, from: &str, to: &str, amount: f64) {
        if let Some(from_bal) = self.faction_balances.get_mut(from) {
            *from_bal -= amount;
        }
        if let Some(to_bal) = self.faction_balances.get_mut(to) {
            *to_bal += amount;
        }
        self.total_abundance += amount * 0.05; // Small abundance bonus on trade
        self.transaction_count += 1;
        self.last_update = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    pub fn apply_production(&mut self, faction: &str, amount: f64) {
        if let Some(balance) = self.faction_balances.get_mut(faction) {
            *balance += amount;
        }
        self.total_abundance += amount;
        self.transaction_count += 1;
        self.last_update = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    }

    // Mercy-relevant metrics
    pub fn mercy_metrics(&self) -> serde_json::Value {
        json!({
            "total_abundance": self.total_abundance,
            "transaction_count": self.transaction_count,
            "faction_count": self.faction_balances.len(),
            "average_balance": if !self.faction_balances.is_empty() {
                self.faction_balances.values().sum::<f64>() / self.faction_balances.len() as f64
            } else { 0.0 }
        })
    }
}

// Shared RBE State
type SharedRbeState = Arc<RwLock<RbeState>>;

// ... (rest of previous code: LogLevel, Event, load_config, AtomicConfig, etc.) ...

fn main() {
    // ... previous initialization ...

    let rbe_state: SharedRbeState = Arc::new(RwLock::new(RbeState::new()));

    // Pass rbe_state into the event loop or make it accessible
    // For simplicity in this version, we'll use a global for demo purposes
    // In production we would pass it properly through the system

    // ... existing main loop with RBE events ...

    // Example integration point (inside RbeTransaction handler):
    // if let Ok(mut state) = rbe_state.write() {
    //     state.apply_transaction(&from_faction, &to_faction, amount);
    //     log_structured(LogLevel::Info, "RBE State Updated", state.mercy_metrics());
    // }

    println!("[Powrush Server] Shared RBE state with mercy metrics active");
}

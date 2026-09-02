//! powrush/src/common/mod.rs
//! Shared types and logic for Powrush (server + client)

use std::collections::HashMap;
use serde_json::json;

#[derive(Debug, Clone, Default)]
pub struct RbeState {
    pub faction_balances: HashMap<String, f64>,
    pub total_abundance: f64,
    pub transaction_count: u64,
    pub last_update: u64,
}

impl RbeState {
    pub fn new() -> Self {
        let mut balances = HashMap::new();
        balances.insert("Sovereign".to_string(), 10000.0);
        balances.insert("Harvesters".to_string(), 8000.0);
        balances.insert("Guardians".to_string(), 6000.0);
        balances.insert("Innovators".to_string(), 7000.0);
        balances.insert("Nomads".to_string(), 5000.0);

        Self {
            faction_balances: balances,
            total_abundance: 36000.0,
            transaction_count: 0,
            last_update: 0,
        }
    }

    pub fn apply_transaction(&mut self, from: &str, to: &str, amount: f64) {
        if let Some(from_bal) = self.faction_balances.get_mut(from) {
            *from_bal -= amount;
        }
        if let Some(to_bal) = self.faction_balances.get_mut(to) {
            *to_bal += amount;
        }
        self.total_abundance += amount * 0.05;
        self.transaction_count += 1;
        self.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

    pub fn apply_production(&mut self, faction: &str, amount: f64) {
        if let Some(balance) = self.faction_balances.get_mut(faction) {
            *balance += amount;
        }
        self.total_abundance += amount;
        self.transaction_count += 1;
        self.last_update = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }

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
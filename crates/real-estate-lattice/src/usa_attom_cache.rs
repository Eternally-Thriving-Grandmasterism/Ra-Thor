//! ATTOM API Caching Strategy for RREL
//! Simple, effective, mercy-aware caching layer for external real estate data providers.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct CacheEntry<T> {
    value: T,
    inserted_at: Instant,
    ttl: Duration,
}

impl<T> CacheEntry<T> {
    fn is_expired(&self) -> bool {
        self.inserted_at.elapsed() > self.ttl
    }
}

pub struct AttomCache {
    property_profiles: DashMap<String, CacheEntry<PropertyProfile>>,
    risk_signals: DashMap<String, CacheEntry<RiskSignals>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyProfile {
    pub parcel_id: String,
    pub owner_name: Option<String>,
    pub tax_assessed_value: Option<f64>,
    pub last_sale_price: Option<f64>,
    pub last_sale_date: Option<String>,
    pub climate_risk_score: Option<f64>,
    pub environmental_flags: Vec<String>,
    pub data_source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskSignals {
    pub flood_risk: Option<String>,
    pub wildfire_risk: Option<String>,
    pub earthquake_risk: Option<String>,
    pub overall_risk_score: Option<f64>,
    pub data_source: String,
}

impl AttomCache {
    pub fn new() -> Self {
        Self {
            property_profiles: DashMap::new(),
            risk_signals: DashMap::new(),
        }
    }

    fn make_key(state: &str, identifier: &str, data_type: &str) -> String {
        format!("{}:{}:{}", state.to_uppercase(), identifier, data_type)
    }

    pub fn get_property_profile(&self, state: &str, identifier: &str) -> Option<PropertyProfile> {
        let key = Self::make_key(state, identifier, "profile");
        if let Some(entry) = self.property_profiles.get(&key) {
            if !entry.is_expired() {
                return Some(entry.value.clone());
            } else {
                self.property_profiles.remove(&key);
            }
        }
        None
    }

    pub fn insert_property_profile(&self, state: &str, identifier: &str, profile: PropertyProfile, ttl_seconds: u64) {
        let key = Self::make_key(state, identifier, "profile");
        let entry = CacheEntry {
            value: profile,
            inserted_at: Instant::now(),
            ttl: Duration::from_secs(ttl_seconds),
        };
        self.property_profiles.insert(key, entry);
    }

    pub fn get_risk_signals(&self, state: &str, identifier: &str) -> Option<RiskSignals> {
        let key = Self::make_key(state, identifier, "risk");
        if let Some(entry) = self.risk_signals.get(&key) {
            if !entry.is_expired() {
                return Some(entry.value.clone());
            } else {
                self.risk_signals.remove(&key);
            }
        }
        None
    }

    pub fn insert_risk_signals(&self, state: &str, identifier: &str, signals: RiskSignals, ttl_seconds: u64) {
        let key = Self::make_key(state, identifier, "risk");
        let entry = CacheEntry {
            value: signals,
            inserted_at: Instant::now(),
            ttl: Duration::from_secs(ttl_seconds),
        };
        self.risk_signals.insert(key, entry);
    }

    pub fn clear(&self) {
        self.property_profiles.clear();
        self.risk_signals.clear();
    }
}

impl Default for AttomCache {
    fn default() -> Self {
        Self::new()
    }
}
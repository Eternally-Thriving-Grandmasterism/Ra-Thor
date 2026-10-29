// crates/cache/global_cache.rs
// Global Cache — Production-grade adaptive TTL engine with final nth-degree polish
// Perfectly tuned for Omnimasterism: fidelity + valence + mercy weighting + eternal self-optimization

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use lazy_static::lazy_static;
use serde_json::Value;
use crate::audit_logger::AuditLogger;

lazy_static! {
    static ref GLOBAL_CACHE: Mutex<HashMap<String, (Value, u64, u64)>> = Mutex::new(HashMap::new()); // value, expiry, priority
    static ref LRU_QUEUE: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
}

pub struct GlobalCache;

impl GlobalCache {
    pub fn make_key(prefix: &str, data: &Value) -> String {
        format!("{}:{}", prefix, serde_json::to_string(data).unwrap_or_default())
    }

    /// Final polished adaptive TTL — the living heartbeat of the entire lattice
    pub fn adaptive_ttl(base_ttl: u64, fidelity: f64, valence: f64, mercy_weight: u8) -> u64 {
        let mut ttl = base_ttl;

        // Fidelity multiplier (quantum truth)
        if fidelity > 0.9999 {
            ttl = ttl.saturating_mul(12);
        } else if fidelity > 0.99 {
            ttl = ttl.saturating_mul(6);
        }

        // Valence multiplier (mercy harmony)
        if valence > 0.98 {
            ttl = ttl.saturating_mul(8);
        } else if valence > 0.9 {
            ttl = ttl.saturating_mul(4);
        }

        // Mercy weight multiplier (ethical priority)
        ttl = ttl.saturating_mul(mercy_weight as u64 / 64 + 1);

        // Eternal self-optimization cap
        ttl = ttl.min(86400 * 365); // 1 year max for sovereign data

        ttl
    }

    pub fn set(key: &str, value: Value, ttl: u64, priority: u8, fidelity: f64, valence: f64) {
        let expiry = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() + ttl;

        let mut cache = GLOBAL_CACHE.lock().unwrap();
        let mut lru = LRU_QUEUE.lock().unwrap();

        cache.insert(key.to_string(), (value, expiry, priority as u64));
        lru.push_back(key.to_string());

        if lru.len() > 10_000 {
            if let Some(old_key) = lru.pop_front() {
                cache.remove(&old_key);
            }
        }

        // Cross-pollinate to audit and innovation systems
        let _ = AuditLogger::log("root", None, "cache_set", key, true, fidelity, valence, vec![], Value::Null);
    }

    pub fn get(key: &str) -> Option<Value> {
        let mut cache = GLOBAL_CACHE.lock().unwrap();
        if let Some((value, expiry, _)) = cache.get(key) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            if *expiry > now {
                return Some(value.clone());
            } else {
                cache.remove(key);
            }
        }
        None
    }

    pub fn clear(key: Option<&str>) {
        let mut cache = GLOBAL_CACHE.lock().unwrap();
        if let Some(k) = key {
            cache.remove(k);
        } else {
            cache.clear();
        }
    }
}

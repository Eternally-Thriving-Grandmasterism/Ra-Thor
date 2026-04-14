// core/global_cache.rs
// Global Cache Module — Single source of truth for all cached results in Ra-Thor
// Used by Master Sovereign Kernel for maximum efficiency and zero redundancy.

use std::collections::HashMap;
use std::sync::Mutex;
use lazy_static::lazy_static;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub value: Value,
    pub timestamp: u64,
    pub ttl_seconds: u64,  // 0 = never expires
}

lazy_static! {
    static ref GLOBAL_CACHE: Mutex<HashMap<String, CacheEntry>> = Mutex::new(HashMap::new());
}

pub struct GlobalCache;

impl GlobalCache {
    /// Get cached value if still valid
    pub fn get(key: &str) -> Option<Value> {
        let cache = GLOBAL_CACHE.lock().unwrap();
        if let Some(entry) = cache.get(key) {
            if entry.ttl_seconds == 0 || Self::is_valid(&entry) {
                return Some(entry.value.clone());
            }
        }
        None
    }

    /// Set or update cached value
    pub fn set(key: &str, value: Value, ttl_seconds: u64) {
        let mut cache = GLOBAL_CACHE.lock().unwrap();
        let entry = CacheEntry {
            value,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ttl_seconds,
        };
        cache.insert(key.to_string(), entry);
    }

    fn is_valid(entry: &CacheEntry) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        now < entry.timestamp + entry.ttl_seconds
    }

    /// Clear specific key or entire cache
    pub fn clear(key: Option<&str>) {
        let mut cache = GLOBAL_CACHE.lock().unwrap();
        if let Some(k) = key {
            cache.remove(k);
        } else {
            cache.clear();
        }
    }

    /// Generate cache key for FENCA/Mermin/mercy operations
    pub fn make_key(prefix: &str, request_data: &Value) -> String {
        format!("{}:{}", prefix, serde_json::to_string(request_data).unwrap_or_default())
    }
}

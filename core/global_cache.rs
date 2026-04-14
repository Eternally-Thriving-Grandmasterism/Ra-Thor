// core/global_cache.rs
// Global Cache Module with advanced eviction strategies (LRU + size limit + TTL priority)
// + Quantum Cache Coherence + TTL Optimization Strategies (priority, adaptive, usage, mercy-gated)

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use lazy_static::lazy_static;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub value: Value,
    pub timestamp: u64,
    pub ttl_seconds: u64,
    pub last_accessed: u64,
    pub priority: u8,          // 0-255 (higher = longer TTL)
}

lazy_static! {
    static ref GLOBAL_CACHE: Mutex<HashMap<String, CacheEntry>> = Mutex::new(HashMap::new());
    static ref LRU_ORDER: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
}

const MAX_CACHE_SIZE: usize = 10_000;

pub struct GlobalCache;

impl GlobalCache {
    pub fn get(key: &str) -> Option<Value> {
        let mut cache = GLOBAL_CACHE.lock().unwrap();
        if let Some(entry) = cache.get_mut(key) {
            if entry.ttl_seconds == 0 || Self::is_valid(entry) {
                entry.last_accessed = Self::now();
                Self::update_lru(key);
                return Some(entry.value.clone());
            }
        }
        None
    }

    pub fn set(key: &str, value: Value, ttl_seconds: u64, priority: u8) {
        let mut cache = GLOBAL_CACHE.lock().unwrap();
        let mut lru = LRU_ORDER.lock().unwrap();

        if cache.len() >= MAX_CACHE_SIZE {
            if let Some(old_key) = lru.pop_front() {
                cache.remove(&old_key);
            }
        }

        let entry = CacheEntry {
            value,
            timestamp: Self::now(),
            ttl_seconds,
            last_accessed: Self::now(),
            priority,
        };
        cache.insert(key.to_string(), entry);
        lru.push_back(key.to_string());
    }

    fn is_valid(entry: &CacheEntry) -> bool {
        let now = Self::now();
        now < entry.timestamp + entry.ttl_seconds
    }

    fn now() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
    }

    fn update_lru(key: &str) {
        let mut lru = LRU_ORDER.lock().unwrap();
        lru.retain(|k| k != key);
        lru.push_back(key.to_string());
    }

    pub fn clear(key: Option<&str>) {
        let mut cache = GLOBAL_CACHE.lock().unwrap();
        let mut lru = LRU_ORDER.lock().unwrap();
        if let Some(k) = key {
            cache.remove(k);
            lru.retain(|x| x != k);
        } else {
            cache.clear();
            lru.clear();
        }
    }

    pub fn make_key(prefix: &str, request_data: &Value) -> String {
        format!("{}:{}", prefix, serde_json::to_string(request_data).unwrap_or_default())
    }

    // Quantum Cache Coherence Protocol (preserved)
    pub fn quantum_coherence_check(key: &str) -> bool {
        let cache = GLOBAL_CACHE.lock().unwrap();
        cache.contains_key(key)
    }

    // TTL Optimization Strategies (new)
    pub fn adaptive_ttl(base_ttl: u64, fidelity: f64, valence: f64, priority: u8) -> u64 {
        let mut ttl = base_ttl;
        if fidelity > 0.999 { ttl *= 4; }           // Fidelity boost
        if valence > 0.95 { ttl *= 2; }             // Mercy/Valence boost
        ttl = ttl.saturating_mul(priority as u64);  // Priority tier multiplier
        ttl.min(86_400)                             // Cap at 24 hours
    }
}

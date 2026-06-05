//! powrush/src/server/main.rs
//! Headless Powrush Server — Lock-Free Atomic Config Swapping with arc-swap (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Read;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};

use arc_swap::ArcSwap;
use serde::Deserialize;
use serde_json::json;

// ... (Event, LogLevel, LogEntry definitions remain the same) ...

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerConfig {
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_flush_interval_ms")]
    pub flush_interval_ms: u64,
}

fn default_batch_size() -> usize { 128 }
fn default_flush_interval_ms() -> u64 { 100 }

fn load_config() -> ServerConfig {
    let path = "powrush_config.json";
    if let Ok(mut file) = std::fs::File::open(path) {
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_ok() {
            if let Ok(config) = serde_json::from_str::<ServerConfig>(&contents) {
                return config;
            }
        }
    }
    ServerConfig { batch_size: 128, flush_interval_ms: 100 }
}

// Lock-free atomic config using arc-swap
type AtomicConfig = ArcSwap<ServerConfig>;

fn start_config_watcher(atomic_config: std::sync::Arc<AtomicConfig>) {
    thread::spawn(move || {
        let config_path = "powrush_config.json";
        let mut last_modified = None;

        loop {
            if let Ok(metadata) = std::fs::metadata(config_path) {
                let modified = metadata.modified().ok();

                if last_modified != modified {
                    last_modified = modified;
                    let new_config = load_config();

                    // Lock-free atomic swap
                    atomic_config.store(std::sync::Arc::new(new_config.clone()));

                    log_structured(LogLevel::Info, "Config hot-reloaded (lock-free atomic swap)", json!({
                        "batch_size": new_config.batch_size,
                        "flush_interval_ms": new_config.flush_interval_ms
                    }));
                }
            }
            thread::sleep(Duration::from_secs(2));
        }
    });
}

fn init_log_batcher(atomic_config: std::sync::Arc<AtomicConfig>) -> mpsc::Sender<LogEntry> {
    let (tx, rx) = mpsc::channel::<LogEntry>();

    thread::spawn(move || {
        let mut batch: Vec<LogEntry> = Vec::new();
        let mut last_flush = Instant::now();

        loop {
            // Lock-free load of current config
            let config = atomic_config.load();
            let current_batch_size = config.batch_size;
            let current_flush_ms = config.flush_interval_ms;

            if let Ok(entry) = rx.recv_timeout(Duration::from_millis(30)) {
                batch.push(entry);
            }

            let should_flush = batch.len() >= current_batch_size 
                || last_flush.elapsed() >= Duration::from_millis(current_flush_ms);

            if should_flush && !batch.is_empty() {
                flush_batch(&batch);
                batch.clear();
                last_flush = Instant::now();
            }
        }
    });

    tx
}

// ... (flush_batch, log_structured, log_mercy_json, log_error, evaluate_mercy remain similar) ...

fn main() {
    let initial_config = load_config();
    let atomic_config = std::sync::Arc::new(AtomicConfig::from(initial_config.clone()));

    let sender = init_log_batcher(std::sync::Arc::clone(&atomic_config));
    // (LOG_SENDER setup for compatibility)

    start_config_watcher(std::sync::Arc::clone(&atomic_config));

    log_structured(LogLevel::Info, "Powrush Server starting with lock-free atomic config (arc-swap)", json!({
        "batch_size": initial_config.batch_size,
        "flush_interval_ms": initial_config.flush_interval_ms
    }));

    // ... rest of main loop unchanged ...

    println!("[Powrush Server] Using lock-free atomic config swapping via arc-swap");
}

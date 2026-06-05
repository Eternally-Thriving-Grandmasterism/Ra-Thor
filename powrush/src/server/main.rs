//! powrush/src/server/main.rs
//! Headless Powrush Server — Configurable Async Log Batching (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::json;

// Default values (used if config file missing or invalid)
const DEFAULT_BATCH_SIZE: usize = 128;
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
    Debug,
}

impl LogLevel {
    fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Debug => "DEBUG",
        }
    }
}

#[derive(Debug, Clone)]
pub enum Event {
    Tick { tick: u64 },
    DiplomacyProposal { from: String, to: String, proposal_type: String },
    CouncilModulation { council_id: u8, action: String },
    EvolutionProposal { module: String, benefit: f64 },
    PlayerAction { player_id: u64, action: String },
    RbeTransaction {
        from_faction: String,
        to_faction: String,
        resource: String,
        amount: f64,
        reason: String,
    },
    ResourceProduction {
        faction: String,
        resource: String,
        amount: f64,
    },
    AbundanceFlow { amount: f64, description: String },
    Shutdown,
}

#[derive(Debug, Clone)]
struct LogEntry {
    level: LogLevel,
    message: String,
    context: serde_json::Value,
}

// Runtime batch configuration
static mut BATCH_SIZE: usize = DEFAULT_BATCH_SIZE;
static mut FLUSH_INTERVAL_MS: u64 = DEFAULT_FLUSH_INTERVAL_MS;
static mut LOG_SENDER: Option<Sender<LogEntry>> = None;

#[derive(Deserialize, Debug, Default)]
struct ServerConfig {
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default = "default_flush_interval_ms")]
    flush_interval_ms: u64,
}

fn default_batch_size() -> usize { DEFAULT_BATCH_SIZE }
fn default_flush_interval_ms() -> u64 { DEFAULT_FLUSH_INTERVAL_MS }

fn load_config() -> ServerConfig {
    let path = "powrush_config.json";

    if let Ok(mut file) = std::fs::File::open(path) {
        let mut contents = String::new();
        if file.read_to_string(&mut contents).is_ok() {
            if let Ok(config) = serde_json::from_str::<ServerConfig>(&contents) {
                println!("[Config] Loaded batch settings from {}", path);
                return config;
            }
        }
    }

    println!("[Config] Using default batch settings (no valid {} found)", path);
    ServerConfig {
        batch_size: DEFAULT_BATCH_SIZE,
        flush_interval_ms: DEFAULT_FLUSH_INTERVAL_MS,
    }
}

fn init_log_batcher(batch_size: usize, flush_interval_ms: u64) -> Sender<LogEntry> {
    let (tx, rx) = mpsc::channel::<LogEntry>();

    thread::spawn(move || {
        let mut batch: Vec<LogEntry> = Vec::with_capacity(batch_size);
        let mut last_flush = std::time::Instant::now();

        for entry in rx {
            batch.push(entry);

            if batch.len() >= batch_size || last_flush.elapsed() >= Duration::from_millis(flush_interval_ms) {
                flush_batch(&batch);
                batch.clear();
                last_flush = std::time::Instant::now();
            }
        }

        if !batch.is_empty() {
            flush_batch(&batch);
        }
    });

    tx
}

fn flush_batch(batch: &[LogEntry]) {
    if batch.is_empty() { return; }

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("powrush_structured_log.jsonl")
    {
        for entry in batch {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let json_entry = json!({
                "timestamp": timestamp,
                "level": entry.level.as_str(),
                "message": entry.message,
                "context": entry.context
            });

            let line = json_entry.to_string() + "\n";
            let _ = file.write_all(line.as_bytes());
        }
    }
}

fn log_structured(level: LogLevel, message: &str, context: serde_json::Value) {
    unsafe {
        if let Some(sender) = &LOG_SENDER {
            let _ = sender.send(LogEntry {
                level,
                message: message.to_string(),
                context,
            });
        }
    }
}

fn log_mercy_json(entry: serde_json::Value) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("powrush_mercy_audit.jsonl")
    {
        let line = entry.to_string() + "\n";
        let _ = file.write_all(line.as_bytes());
    }
}

fn log_error(entry: serde_json::Value) {
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("powrush_error_trace.jsonl")
    {
        let line = entry.to_string() + "\n";
        let _ = file.write_all(line.as_bytes());
    }
}

fn evaluate_mercy(event: &Event) -> bool {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let (decision, gate_failed, details) = match event {
        Event::RbeTransaction { amount, reason, .. } => {
            if *amount < 0.0 { ("REJECTED", "Gate 1 (Non-Harm)", json!({"amount": amount})) }
            else if reason.to_lowercase().contains("exploit") || reason.to_lowercase().contains("hoard") { ("REJECTED", "Gate 4/5 (Abundance/Truth)", json!({"reason": reason})) }
            else if reason.len() < 10 { ("REJECTED", "Gate 5 (Truth)", json!({"reason_length": reason.len()})) }
            else { ("PASSED", "", json!({})) }
        }
        Event::AbundanceFlow { amount, .. } => {
            if *amount <= 0.0 { ("REJECTED", "Gate 4/7 (Abundance/Harmony)", json!({"amount": amount})) }
            else { ("PASSED", "", json!({})) }
        }
        Event::EvolutionProposal { benefit, .. } => {
            if *benefit < 0.75 { ("REJECTED", "Gate 2/5 (Mercy/Truth)", json!({"benefit": benefit})) }
            else { ("PASSED", "", json!({})) }
        }
        Event::DiplomacyProposal { proposal_type, .. } => {
            if proposal_type.to_lowercase().contains("war") || proposal_type.to_lowercase().contains("dominate") { ("REJECTED", "Gate 3/6 (Service/Joy)", json!({"proposal_type": proposal_type})) }
            else { ("PASSED", "", json!({})) }
        }
        _ => ("PASSED", "", json!({})),
    };

    let audit_entry = json!({
        "timestamp": timestamp,
        "event_type": format!("{:?}", event).split('(').next().unwrap_or("Unknown"),
        "decision": decision,
        "gate_failed": gate_failed,
        "details": details
    });

    log_mercy_json(audit_entry);
    decision == "PASSED"
}

fn main() {
    // Load configuration
    let config = load_config();

    // Apply loaded config to statics
    unsafe {
        BATCH_SIZE = config.batch_size;
        FLUSH_INTERVAL_MS = config.flush_interval_ms;
    }

    let sender = init_log_batcher(config.batch_size, config.flush_interval_ms);
    unsafe { LOG_SENDER = Some(sender); }

    log_structured(LogLevel::Info, "Powrush Server starting with configurable batching", json!({
        "batch_size": config.batch_size,
        "flush_interval_ms": config.flush_interval_ms,
        "config_file": "powrush_config.json"
    }));

    let mut organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();

    let mut diplomacy = FactionDiplomacy::new();
    let mut evolution_gate = SelfEvolutionGate::new();

    let mut event_queue: VecDeque<Event> = VecDeque::new();
    event_queue.push_back(Event::Tick { tick: 0 });
    event_queue.push_back(Event::RbeTransaction {
        from_faction: "Harvesters".to_string(),
        to_faction: "Sovereigns".to_string(),
        resource: "Biomaterial".to_string(),
        amount: 1250.0,
        reason: "Initial resource allocation for sovereign construction".to_string(),
    });

    let mut current_tick: u64 = 0;
    let max_events = 25;

    log_structured(LogLevel::Info, "Event loop started", json!({ "max_events": max_events }));

    while let Some(event) = event_queue.pop_front() {
        if !evaluate_mercy(&event) { continue; }

        match event {
            Event::Tick { tick } => {
                current_tick = tick;
                log_structured(LogLevel::Debug, "Processing tick", json!({ "tick": current_tick }));
                organism.offer_cosmic_loop();

                if current_tick < 30 {
                    event_queue.push_back(Event::Tick { tick: current_tick + 1 });
                }

                if current_tick % 5 == 0 {
                    event_queue.push_back(Event::ResourceProduction {
                        faction: "Harvesters".to_string(),
                        resource: "Energy".to_string(),
                        amount: 750.0,
                    });
                }
            }

            Event::RbeTransaction { from_faction, to_faction, resource, amount, reason } => {
                log_structured(LogLevel::Info, "RBE Transaction processed", json!({
                    "from": from_faction, "to": to_faction, "resource": resource, "amount": amount
                }));

                if let Err(e) = diplomacy.propose_diplomacy(powrush::DiplomacyProposal {
                    id: current_tick,
                    from: powrush::Faction::Harvesters,
                    to: powrush::Faction::Sovereigns,
                    proposal_type: "Resource Trade".to_string(),
                    terms: reason.clone(),
                    mercy_impact: 0.999,
                    rbe_value: amount,
                    evolution_potential: if amount > 1000.0 { 0.85 } else { 0.6 },
                }) {
                    log_error(json!({
                        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        "event_type": "RbeTransaction",
                        "component": "FactionDiplomacy",
                        "error": e.to_string(),
                        "context": { "from": from_faction, "to": to_faction, "amount": amount }
                    }));
                }

                let abundance_threshold = 800.0;
                if amount > abundance_threshold {
                    event_queue.push_back(Event::AbundanceFlow {
                        amount: (amount - abundance_threshold) * 0.15,
                        description: format!("Overflow from {} transfer", resource),
                    });
                }

                if amount > 2000.0 {
                    event_queue.push_back(Event::EvolutionProposal {
                        module: format!("rbe_{}", resource.to_lowercase()),
                        benefit: 0.91,
                    });
                }
            }

            Event::ResourceProduction { faction, resource, amount } => {
                log_structured(LogLevel::Debug, "Resource production", json!({
                    "faction": faction, "resource": resource, "amount": amount
                }));
            }

            Event::AbundanceFlow { amount, description } => {
                log_structured(LogLevel::Info, "Abundance flow generated", json!({
                    "amount": amount, "description": description
                }));

                if amount > 120.0 {
                    event_queue.push_back(Event::EvolutionProposal {
                        module: "global_abundance".to_string(),
                        benefit: 0.89,
                    });
                }
            }

            Event::DiplomacyProposal { from, to, proposal_type } => {
                log_structured(LogLevel::Info, "Diplomacy proposal processed", json!({
                    "from": from, "to": to, "proposal_type": proposal_type
                }));
            }

            Event::CouncilModulation { council_id, action } => {
                log_structured(LogLevel::Info, "Council modulation", json!({
                    "council_id": council_id, "action": action
                }));
            }

            Event::EvolutionProposal { module, benefit } => {
                log_structured(LogLevel::Info, "Evolution proposal processed", json!({
                    "module": module, "benefit": benefit
                }));

                let proposal = powrush::EvolutionProposal {
                    id: current_tick + 5000,
                    proposer: "RBE_Chain".to_string(),
                    target_module: module.clone(),
                    description: format!("RBE evolution benefit {:.2}", benefit),
                    proposed_diff: "Improve economic flow".to_string(),
                    expected_benefit: benefit,
                    risk_score: 0.0001,
                    mercy_alignment: 0.999,
                };

                if let Err(e) = evolution_gate.propose_evolution(proposal) {
                    log_error(json!({
                        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        "event_type": "EvolutionProposal",
                        "component": "SelfEvolutionGate",
                        "error": e.to_string(),
                        "context": { "module": module, "benefit": benefit }
                    }));
                }
            }

            Event::PlayerAction { player_id, action } => {
                log_structured(LogLevel::Info, "Player action received", json!({
                    "player_id": player_id, "action": action
                }));
            }

            Event::Shutdown => {
                log_structured(LogLevel::Info, "Server shutting down", json!({}));
                break;
            }
        }

        thread::sleep(Duration::from_millis(50));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    log_structured(LogLevel::Info, "Simulation complete", json!({}));
    println!("[Powrush Server] Using batch config: size={} / interval={}ms", unsafe { BATCH_SIZE }, unsafe { FLUSH_INTERVAL_MS });
    println!("[Powrush Server] Config file: powrush_config.json (optional)");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

// Note: load_config function is defined above

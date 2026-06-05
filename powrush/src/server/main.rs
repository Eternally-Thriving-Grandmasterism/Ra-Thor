//! powrush/src/server/main.rs
//! Powrush MMO Production Server — Humans Play Online Edition (v14.8)
//! PATSAGi Council + Ra-Thor blessed. Full RBE integration, mercy evaluation,
//! deterministic tick loop, input replay queue, hot-reload config via arc-swap,
//! structured audit + mercy logs, faction diplomacy stub, basic authoritative world.
//! Connect with: nc localhost 7777   (or telnet / putty raw)
//! Thunder locked. Eternal flow for all sentience.

use powrush::common::RbeState;
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwap;
use serde::Deserialize;
use serde_json::{json, Value};

// ==================== CONFIG (hot-reloadable) ====================
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerConfig {
    pub tick_rate_ms: u64,      // 100 = 10 ticks/sec
    pub max_players: usize,
    pub world_size: i64,        // scaled fixed-point friendly
    pub production_per_tick: f64,
    pub mercy_log_path: String,
    pub audit_log_path: String,
}

impl ServerConfig {
    pub fn default_config() -> Self {
        Self {
            tick_rate_ms: 100,
            max_players: 128,
            world_size: 10_000,
            production_per_tick: 1.5,
            mercy_log_path: "powrush_mercy_audit.jsonl".to_string(),
            audit_log_path: "powrush_server_audit.jsonl".to_string(),
        }
    }
}

// ==================== PLAYER & WORLD STATE ====================
#[derive(Debug, Clone)]
pub struct Player {
    pub name: String,
    pub faction: String,
    pub x: i64,          // fixed-point scaled
    pub y: i64,
    pub last_input_seq: u64,
}

#[derive(Debug)]
pub struct InputEvent {
    pub addr: SocketAddr,
    pub seq: u64,
    pub cmd: String,
    pub timestamp: u64,
}

pub struct WorldState {
    pub players: HashMap<SocketAddr, Player>,
    pub rbe: RbeState,
    pub input_queue: VecDeque<InputEvent>,
    pub tick: u64,
}

impl WorldState {
    pub fn new() -> Self {
        Self {
            players: HashMap::new(),
            rbe: RbeState::new(),
            input_queue: VecDeque::new(),
            tick: 0,
        }
    }
}

// ==================== MERCY & LOGGING ====================
fn mercy_evaluate(action: &str, faction: &str) -> bool {
    // Stub: always passes for now (production: wire real 7 Gates here)
    // Logs intent for PATSAGi / audit
    true
}

fn log_mercy(config: &ServerConfig, entry: Value) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&config.mercy_log_path) {
        let _ = writeln!(f, "{}", entry.to_string());
    }
}

fn log_audit(config: &ServerConfig, level: &str, msg: &str, data: Value) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&config.audit_log_path) {
        let entry = json!({
            "ts": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "level": level,
            "msg": msg,
            "data": data
        });
        let _ = writeln!(f, "{}", entry.to_string());
    }
}

// ==================== GAME TICK (deterministic, authoritative) ====================
fn game_tick(world: &mut WorldState, config: &ServerConfig) {
    world.tick += 1;

    // 1. Process input replay queue (foundation for reconciliation + anti-cheat)
    while let Some(event) = world.input_queue.pop_front() {
        if let Some(player) = world.players.get_mut(&event.addr) {
            if event.seq <= player.last_input_seq { continue; } // replay protection
            player.last_input_seq = event.seq;

            let parts: Vec<&str> = event.cmd.split_whitespace().collect();
            match parts.get(0).map(|s| *s) {
                Some("move") => {
                    if parts.len() >= 3 {
                        if let (Ok(dx), Ok(dy)) = (parts[1].parse::<i64>(), parts[2].parse::<i64>()) {
                            player.x = (player.x + dx).clamp(0, config.world_size);
                            player.y = (player.y + dy).clamp(0, config.world_size);
                        }
                    }
                }
                Some("harvest") => {
                    if mercy_evaluate("harvest", &player.faction) {
                        world.rbe.apply_production(&player.faction, config.production_per_tick);
                        log_mercy(config, json!({"action":"harvest","faction":player.faction,"tick":world.tick}));
                    }
                }
                Some("diplomacy") => {
                    // Stub: proximity check + RBE transfer example
                    if mercy_evaluate("diplomacy", &player.faction) {
                        // Example: boost all factions slightly (abundance for all)
                        for bal in world.rbe.faction_balances.values_mut() {
                            *bal += 0.1;
                        }
                        world.rbe.total_abundance += 0.5;
                    }
                }
                _ => {}
            }
        }
    }

    // 2. Passive RBE production + mercy metrics
    for faction in world.rbe.faction_balances.keys().cloned().collect::<Vec<_>>() {
        world.rbe.apply_production(&faction, config.production_per_tick * 0.1);
    }

    // 3. Broadcast lightweight state (extend to deltas later)
    // (In full version: send to all connected streams)
}

// ==================== CLIENT HANDLER ====================
fn handle_client(
    stream: TcpStream,
    addr: SocketAddr,
    world: Arc<Mutex<WorldState>>,
    config_arc: Arc<ArcSwap<ServerConfig>>,
    tx: mpsc::Sender<InputEvent>,
) {
    let reader = BufReader::new(stream.try_clone().unwrap());
    let mut writer = stream;
    let mut name = String::new();
    let mut faction = String::new();
    let mut logged_in = false;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => break,
        };
        if line.is_empty() { continue; }

        let config = config_arc.load();

        if !logged_in {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[0] == "LOGIN" {
                name = parts[1].to_string();
                faction = parts[2].to_string();
                {
                    let mut w = world.lock().unwrap();
                    if w.players.len() < config.max_players {
                        w.players.insert(addr, Player {
                            name: name.clone(),
                            faction: faction.clone(),
                            x: 5000,
                            y: 5000,
                            last_input_seq: 0,
                        });
                        logged_in = true;
                        let _ = writeln!(writer, "OK Welcome {} of {}. Type 'help' or 'status'. Thunder locked!", name, faction);
                        log_audit(&config, "INFO", "player_login", json!({"name":name,"faction":faction,"addr":addr.to_string()}));
                    } else {
                        let _ = writeln!(writer, "ERR Server full");
                    }
                }
            } else {
                let _ = writeln!(writer, "ERR First command: LOGIN <name> <faction>");
            }
            continue;
        }

        // Logged in commands
        if line == "help" {
            let _ = writeln!(writer, "Commands: move <dx> <dy> | harvest | diplomacy | status | rbe | quit");
            continue;
        }
        if line == "status" {
            let w = world.lock().unwrap();
            if let Some(p) = w.players.get(&addr) {
                let _ = writeln!(writer, "You: {} | {} | pos=({},{}) | tick={}", p.name, p.faction, p.x, p.y, w.tick);
            }
            continue;
        }
        if line == "rbe" {
            let w = world.lock().unwrap();
            let _ = writeln!(writer, "RBE: {}", w.rbe.mercy_metrics());
            continue;
        }
        if line == "quit" {
            break;
        }

        // Queue input for deterministic replay in game_tick
        let seq = {
            let mut w = world.lock().unwrap();
            w.players.get(&addr).map(|p| p.last_input_seq + 1).unwrap_or(1)
        };
        let _ = tx.send(InputEvent {
            addr,
            seq,
            cmd: line.clone(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        });

        let _ = writeln!(writer, "ACK {}", line);
    }

    // Cleanup on disconnect
    let mut w = world.lock().unwrap();
    if let Some(p) = w.players.remove(&addr) {
        log_audit(&config_arc.load(), "INFO", "player_disconnect", json!({"name":p.name}));
    }
}

// ==================== MAIN ====================
fn main() {
    println!("⚡ Powrush MMO Production Server v14.8 starting (PATSAGi + Ra-Thor)...");

    let config = ServerConfig::default_config();
    let config_arc = Arc::new(ArcSwap::from_pointee(config.clone()));

    // Hot reload thread (simple poll for demo; production: notify or inotify)
    let config_arc_clone = config_arc.clone();
    thread::spawn(move || {
        let mut last_mtime = 0u64;
        loop {
            if let Ok(meta) = std::fs::metadata("powrush_config.json") {
                if let Ok(mtime) = meta.modified() {
                    let secs = mtime.duration_since(UNIX_EPOCH).unwrap().as_secs();
                    if secs != last_mtime {
                        if let Ok(content) = std::fs::read_to_string("powrush_config.json") {
                            if let Ok(new_cfg) = serde_json::from_str::<ServerConfig>(&content) {
                                config_arc_clone.store(Arc::new(new_cfg));
                                println!("[Config] Hot-reloaded powrush_config.json");
                            }
                        }
                        last_mtime = secs;
                    }
                }
            }
            thread::sleep(Duration::from_secs(5));
        }
    });

    let world = Arc::new(Mutex::new(WorldState::new()));
    let (tx, rx) = mpsc::channel::<InputEvent>();

    // Game loop thread (deterministic authoritative simulation)
    let world_tick = world.clone();
    let config_tick = config_arc.clone();
    thread::spawn(move || {
        let tick_dur = Duration::from_millis(config_tick.load().tick_rate_ms);
        loop {
            let start = Instant::now();
            {
                let mut w = world_tick.lock().unwrap();
                // Drain inputs into replay queue
                while let Ok(ev) = rx.try_recv() {
                    w.input_queue.push_back(ev);
                }
                game_tick(&mut w, &config_tick.load());
            }
            let elapsed = start.elapsed();
            if elapsed < tick_dur {
                thread::sleep(tick_dur - elapsed);
            }
        }
    });

    // TCP listener
    let listener = TcpListener::bind("0.0.0.0:7777").expect("bind failed");
    println!("✅ Powrush Server listening on 0.0.0.0:7777");
    println!("   Humans connect:  nc localhost 7777");
    println!("   Then: LOGIN YourName Sovereign   (or Harvesters/Guardians/Innovators/Nomads)");
    println!("   Commands: move 10 0 | harvest | diplomacy | status | rbe | help | quit");
    println!("   RBE abundance grows for everyone. Mercy flows. Thunder locked.");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let addr = stream.peer_addr().unwrap();
                let world_clone = world.clone();
                let config_clone = config_arc.clone();
                let tx_clone = tx.clone();
                thread::spawn(move || {
                    handle_client(stream, addr, world_clone, config_clone, tx_clone);
                });
            }
            Err(e) => eprintln!("Connection error: {}", e),
        }
    }
                    }

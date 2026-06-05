//! powrush/src/server/main.rs
//! Powrush MMO Production Server — Humans Play Online Edition (v14.10 WebSocket + TCP)
//! PATSAGi Council + Ra-Thor blessed. Full RBE integration, mercy evaluation,
//! deterministic tick loop, input replay queue, hot-reload config via arc-swap,
//! structured audit + mercy logs, faction diplomacy stub, authoritative world.
//! TCP: nc localhost 7777 (line protocol) | WebSocket: ws://localhost:7778 (JSON for browser)
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

// WebSocket production support (tokio + tokio-tungstenite)
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener as TokioTcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

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
    // (In full version: send to all connected streams via broadcast channel)
}

// ==================== CLIENT HANDLER (TCP line protocol - unchanged, fully compatible) ====================
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

// ==================== WEBSOCKET HANDLER (JSON protocol for browser/WebXR clients) ====================
async fn handle_ws_client(
    stream: tokio::net::TcpStream,
    addr: SocketAddr,
    world: Arc<Mutex<WorldState>>,
    config_arc: Arc<ArcSwap<ServerConfig>>,
    tx: mpsc::Sender<InputEvent>,
) {
    let ws_stream = match accept_async(stream).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("WS accept error from {}: {}", addr, e);
            return;
        }
    };
    let (mut write, mut read) = ws_stream.split();

    let mut logged_in = false;
    let mut name = String::new();
    let mut faction = String::new();

    // Welcome
    let _ = write.send(Message::Text(json!({
        "type": "welcome",
        "msg": "⚡ Welcome to Powrush Web Client — Thunder locked eternally! Send LOGIN to begin."
    }).to_string().into())).await;

    while let Some(msg) = read.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(_) => break,
        };
        if let Message::Text(text) = msg {
            let data: Value = match serde_json::from_str(&text) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let cmd = data.get("cmd").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let config = config_arc.load();

            if !logged_in {
                if cmd == "LOGIN" {
                    if let (Some(n_val), Some(f_val)) = (data.get("name"), data.get("faction")) {
                        if let (Some(n), Some(f)) = (n_val.as_str(), f_val.as_str()) {
                            name = n.to_string();
                            faction = f.to_string();
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
                                    let _ = write.send(Message::Text(json!({
                                        "type": "welcome",
                                        "msg": format!("OK Welcome {} of {}. Thunder locked!", name, faction)
                                    }).to_string().into())).await;
                                    log_audit(&config, "INFO", "ws_player_login", json!({"name":name,"faction":faction,"addr":addr.to_string()}));
                                    // Send initial state snapshot
                                    send_current_state(&mut write, &w).await;
                                } else {
                                    let _ = write.send(Message::Text(json!({"type": "error", "msg": "Server full"}).to_string().into())).await;
                                }
                            }
                        }
                    }
                }
                continue;
            }

            // Logged-in commands
            if cmd == "help" {
                let _ = write.send(Message::Text(json!({
                    "type": "ack",
                    "cmd": "help",
                    "msg": "WASD move | SPACE harvest | D diplomacy | status | rbe | Buttons available"
                }).to_string().into())).await;
                continue;
            }
            if cmd == "status" || cmd == "rbe" {
                let w = world.lock().unwrap();
                if cmd == "status" {
                    if let Some(p) = w.players.get(&addr) {
                        let _ = write.send(Message::Text(json!({
                            "type": "ack",
                            "cmd": "status",
                            "msg": format!("You: {} | {} | pos=({},{}) | tick={}", p.name, p.faction, p.x, p.y, w.tick)
                        }).to_string().into())).await;
                    }
                } else {
                    let _ = write.send(Message::Text(json!({
                        "type": "ack",
                        "cmd": "rbe",
                        "rbe": w.rbe.mercy_metrics()
                    }).to_string().into())).await;
                }
                continue;
            }
            if cmd == "quit" {
                break;
            }

            // Queue for game_tick (move/harvest/diplomacy etc.)
            let seq = {
                let mut w = world.lock().unwrap();
                w.players.get(&addr).map(|p| p.last_input_seq + 1).unwrap_or(1)
            };
            let _ = tx.send(InputEvent {
                addr,
                seq,
                cmd: cmd.clone(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            });

            let _ = write.send(Message::Text(json!({
                "type": "ack",
                "cmd": cmd
            }).to_string().into())).await;

            // Push updated state after action (live HUD update)
            {
                let w = world.lock().unwrap();
                send_current_state(&mut write, &w).await;
            }
        }
    }

    // Cleanup
    let mut w = world.lock().unwrap();
    if let Some(p) = w.players.remove(&addr) {
        log_audit(&config_arc.load(), "INFO", "ws_player_disconnect", json!({"name":p.name}));
    }
}

async fn send_current_state(
    write: &mut futures_util::stream::SplitSink<tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>, Message>,
    world: &WorldState,
) {
    let players_json: Vec<Value> = world.players.values()
        .map(|p| json!({"name": p.name, "faction": p.faction, "x": p.x, "y": p.y }))
        .collect();
    let state = json!({
        "type": "state",
        "tick": world.tick,
        "players": players_json,
        "rbe": {
            "total_abundance": world.rbe.total_abundance,
            "faction_balances": world.rbe.faction_balances
        }
    });
    let _ = write.send(Message::Text(state.to_string().into())).await;
}

// ==================== MAIN ====================
fn main() {
    println!("⚡ Powrush MMO Production Server v14.10 starting (PATSAGi + Ra-Thor + WebSocket)...");

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

    // Game loop thread (deterministic authoritative simulation) - unchanged
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

    // TCP listener (port 7777 - terminal / legacy clients)
    let listener = TcpListener::bind("0.0.0.0:7777").expect("bind failed");
    println!("✅ Powrush TCP listening on 0.0.0.0:7777");
    println!("   Terminal humans: nc localhost 7777");
    println!("   Then: LOGIN YourName Sovereign   (or Harvesters/Guardians/Innovators/Nomads)");
    println!("   Commands: move 10 0 | harvest | diplomacy | status | rbe | help | quit");

    // WebSocket listener for browser clients (port 7778 - JSON, works with powrush-client.html)
    let world_ws = world.clone();
    let config_ws = config_arc.clone();
    let tx_ws = tx.clone();
    thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("tokio runtime for WS");
        rt.block_on(async {
            let ws_listener = TokioTcpListener::bind("0.0.0.0:7778").await.expect("ws bind failed");
            println!("✅ Powrush WebSocket listening on 0.0.0.0:7778");
            println!("   Browser humans: open powrush/web/powrush-client.html");
            println!("   Click Connect → Login → WASD move, SPACE harvest, D diplomacy");
            println!("   RBE abundance grows for EVERY faction. Mercy flows. Thunder locked.");

            loop {
                match ws_listener.accept().await {
                    Ok((stream, addr)) => {
                        let world_c = world_ws.clone();
                        let config_c = config_ws.clone();
                        let tx_c = tx_ws.clone();
                        tokio::spawn(async move {
                            handle_ws_client(stream, addr, world_c, config_c, tx_c).await;
                        });
                    }
                    Err(e) => eprintln!("WS listener error: {}", e),
                }
            }
        });
    });

    // TCP accept loop (blocking, unchanged)
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
            Err(e) => eprintln!("TCP Connection error: {}", e),
        }
    }
}
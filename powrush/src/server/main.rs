//! powrush/src/server/main.rs
//! Headless Powrush Server — Structured JSON Mercy Audit Logging (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;

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

/// Writes a structured JSON audit entry (JSON Lines format)
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

/// Evaluates event against 7 Living Mercy Gates with structured JSON audit logging
fn evaluate_mercy(event: &Event) -> bool {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let (decision, gate_failed, details) = match event {
        Event::RbeTransaction { amount, reason, from_faction, to_faction, resource, .. } => {
            if *amount < 0.0 {
                ("REJECTED", "Gate 1 (Non-Harm)", json!({"amount": amount}));
            } else if reason.to_lowercase().contains("exploit") || reason.to_lowercase().contains("hoard") {
                ("REJECTED", "Gate 4/5 (Abundance/Truth)", json!({"reason": reason}));
            } else if reason.len() < 10 {
                ("REJECTED", "Gate 5 (Truth)", json!({"reason_length": reason.len()}));
            } else {
                ("PASSED", "", json!({}));
            }
        }

        Event::AbundanceFlow { amount, .. } => {
            if *amount <= 0.0 {
                ("REJECTED", "Gate 4/7 (Abundance/Harmony)", json!({"amount": amount}));
            } else {
                ("PASSED", "", json!({}));
            }
        }

        Event::EvolutionProposal { benefit, .. } => {
            if *benefit < 0.75 {
                ("REJECTED", "Gate 2/5 (Mercy/Truth)", json!({"benefit": benefit}));
            } else {
                ("PASSED", "", json!({}));
            }
        }

        Event::DiplomacyProposal { proposal_type, .. } => {
            if proposal_type.to_lowercase().contains("war") || proposal_type.to_lowercase().contains("dominate") {
                ("REJECTED", "Gate 3/6 (Service/Joy)", json!({"proposal_type": proposal_type}));
            } else {
                ("PASSED", "", json!({}));
            }
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
    println!("[Powrush Server] Starting with Structured JSON Mercy Audit Logging...");

    log_mercy_json(json!({
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        "event_type": "SERVER_START",
        "decision": "INFO",
        "gate_failed": "",
        "details": {"message": "Powrush Server initialized with JSON audit logging"}
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

    println!("[Powrush Server] Event loop started. Structured audits -> powrush_mercy_audit.jsonl");

    while let Some(event) = event_queue.pop_front() {
        if !evaluate_mercy(&event) {
            continue;
        }

        match event {
            Event::Tick { tick } => {
                current_tick = tick;
                println!("[Event] Tick {}", current_tick);
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
                println!("[RBE] Tx (Mercy Passed): {} → {} | {} x{:.1}", from_faction, to_faction, resource, amount);

                let _ = diplomacy.propose_diplomacy(powrush::DiplomacyProposal {
                    id: current_tick,
                    from: powrush::Faction::Harvesters,
                    to: powrush::Faction::Sovereigns,
                    proposal_type: "Resource Trade".to_string(),
                    terms: reason.clone(),
                    mercy_impact: 0.999,
                    rbe_value: amount,
                    evolution_potential: if amount > 1000.0 { 0.85 } else { 0.6 },
                });

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
                println!("[RBE] Production: {} produced {:.1} {}", faction, amount, resource);
            }

            Event::AbundanceFlow { amount, description } => {
                println!("[RBE] Abundance: +{:.1} | {}", amount, description);

                if amount > 120.0 {
                    event_queue.push_back(Event::EvolutionProposal {
                        module: "global_abundance".to_string(),
                        benefit: 0.89,
                    });
                }
            }

            Event::DiplomacyProposal { from, to, proposal_type } => {
                println!("[Event] Diplomacy (Mercy Passed): {} → {} ({})", from, to, proposal_type);
            }

            Event::CouncilModulation { council_id, action } => {
                println!("[Event] PATSAGi Council {}: {}", council_id, action);
            }

            Event::EvolutionProposal { module, benefit } => {
                println!("[Event] Evolution (Mercy Passed): {} (benefit: {:.2})", module, benefit);

                let proposal = powrush::EvolutionProposal {
                    id: current_tick + 5000,
                    proposer: "RBE_Chain".to_string(),
                    target_module: module,
                    description: format!("RBE evolution benefit {:.2}", benefit),
                    proposed_diff: "Improve economic flow".to_string(),
                    expected_benefit: benefit,
                    risk_score: 0.0001,
                    mercy_alignment: 0.999,
                };
                let _ = evolution_gate.propose_evolution(proposal);
            }

            Event::PlayerAction { player_id, action } => {
                println!("[Event] Player {}: {}", player_id, action);
            }

            Event::Shutdown => {
                log_mercy_json(json!({
                    "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                    "event_type": "SERVER_SHUTDOWN",
                    "decision": "INFO",
                    "gate_failed": "",
                    "details": {"message": "Event loop terminated"}
                }));
                println!("[Event] Shutdown.");
                break;
            }
        }

        thread::sleep(Duration::from_millis(70));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] Structured JSON Mercy Audit simulation complete.");
    println!("[Powrush Server] Audit log: powrush_mercy_audit.jsonl (JSON Lines format)");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

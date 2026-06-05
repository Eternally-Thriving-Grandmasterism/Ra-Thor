//! powrush/src/server/main.rs
//! Headless Powrush Server — Persistent Mercy Audit Logs (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::Write;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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

/// Appends a mercy audit entry to persistent log file
fn log_mercy_audit(entry: &str) {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let log_line = format!("[{}] {}\n", timestamp, entry);

    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("powrush_mercy_audit.log")
    {
        let _ = file.write_all(log_line.as_bytes());
    }
}

/// Evaluates event against 7 Living Mercy Gates with persistent audit logging
fn evaluate_mercy(event: &Event) -> bool {
    let result = match event {
        Event::RbeTransaction { amount, reason, .. } => {
            if *amount < 0.0 {
                log_mercy_audit(&format!("REJECTED | Gate 1 (Non-Harm) | RbeTransaction | amount={}", amount));
                return false;
            }
            if reason.to_lowercase().contains("exploit") || reason.to_lowercase().contains("hoard") {
                log_mercy_audit("REJECTED | Gate 4/5 (Abundance/Truth) | RbeTransaction | exploitative reason");
                return false;
            }
            if reason.len() < 10 {
                log_mercy_audit("REJECTED | Gate 5 (Truth) | RbeTransaction | insufficient reason detail");
                return false;
            }
            true
        }

        Event::AbundanceFlow { amount, .. } => {
            if *amount <= 0.0 {
                log_mercy_audit("REJECTED | Gate 4/7 (Abundance/Harmony) | AbundanceFlow | non-positive");
                return false;
            }
            true
        }

        Event::EvolutionProposal { benefit, .. } => {
            if *benefit < 0.75 {
                log_mercy_audit(&format!("REJECTED | Gate 2/5 | EvolutionProposal | low benefit={}", benefit));
                return false;
            }
            true
        }

        Event::DiplomacyProposal { proposal_type, .. } => {
            if proposal_type.to_lowercase().contains("war") || proposal_type.to_lowercase().contains("dominate") {
                log_mercy_audit("REJECTED | Gate 3/6 (Service/Joy) | DiplomacyProposal | harmful type");
                return false;
            }
            true
        }

        _ => true,
    };

    if result {
        // Optionally log passes for high-value events (commented for performance)
        // log_mercy_audit(&format!("PASSED | {:?}", event));
    }

    result
}

fn main() {
    println!("[Powrush Server] Starting with Persistent Mercy Audit Logs...");
    log_mercy_audit("SERVER_START | Powrush Server initialized with mercy audit logging");

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

    println!("[Powrush Server] Event loop started. Mercy audits written to powrush_mercy_audit.log");

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
                log_mercy_audit("SERVER_SHUTDOWN | Event loop terminated");
                println!("[Event] Shutdown.");
                break;
            }
        }

        thread::sleep(Duration::from_millis(75));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] Simulation with Persistent Mercy Audit Logs complete.");
    println!("[Powrush Server] Audit log: powrush_mercy_audit.log");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

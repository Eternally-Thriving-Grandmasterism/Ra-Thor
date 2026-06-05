//! powrush/src/server/main.rs
//! Headless Powrush Server with Event-Driven Architecture + RBE Transactions (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use std::collections::VecDeque;
use std::thread;
use std::time::Duration;

/// Core simulation events for the Powrush MMO server
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

fn main() {
    println!("[Powrush Server] Starting event-driven simulation with RBE...");

    let mut organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();

    let mut event_queue: VecDeque<Event> = VecDeque::new();

    // Seed initial events
    event_queue.push_back(Event::Tick { tick: 0 });
    event_queue.push_back(Event::DiplomacyProposal {
        from: "Sovereigns".to_string(),
        to: "Guardians".to_string(),
        proposal_type: "Mutual Abundance Pact".to_string(),
    });
    event_queue.push_back(Event::RbeTransaction {
        from_faction: "Harvesters".to_string(),
        to_faction: "Sovereigns".to_string(),
        resource: "Biomaterial".to_string(),
        amount: 1250.0,
        reason: "Initial resource allocation for construction".to_string(),
    });

    let mut current_tick: u64 = 0;
    let max_events = 20;

    println!("[Powrush Server] Entering RBE-aware event-driven loop...");

    while let Some(event) = event_queue.pop_front() {
        match event {
            Event::Tick { tick } => {
                current_tick = tick;
                println!("[Event] Processing Tick {}", current_tick);

                organism.offer_cosmic_loop();

                if current_tick < 25 {
                    event_queue.push_back(Event::Tick { tick: current_tick + 1 });
                }

                if current_tick % 4 == 0 {
                    event_queue.push_back(Event::DiplomacyProposal {
                        from: "Innovators".to_string(),
                        to: "Nomads".to_string(),
                        proposal_type: "Joint Evolution Project".to_string(),
                    });
                }

                // Periodic RBE production event
                if current_tick % 5 == 0 {
                    event_queue.push_back(Event::ResourceProduction {
                        faction: "Harvesters".to_string(),
                        resource: "Energy".to_string(),
                        amount: 800.0,
                    });
                }
            }

            Event::DiplomacyProposal { from, to, proposal_type } => {
                println!("[Event] Diplomacy: {} -> {} ({}) — Mercy gate passed", from, to, proposal_type);
            }

            Event::CouncilModulation { council_id, action } => {
                println!("[Event] PATSAGi Council {}: {}", council_id, action);
            }

            Event::EvolutionProposal { module, benefit } => {
                println!("[Event] Evolution for {} (benefit: {:.4}) — routed to SelfEvolutionGate", module, benefit);
            }

            Event::PlayerAction { player_id, action } => {
                println!("[Event] Player {}: {}", player_id, action);
            }

            Event::RbeTransaction { from_faction, to_faction, resource, amount, reason } => {
                println!(
                    "[RBE] Transaction: {} -> {} | {} x{:.1} | {}",
                    from_faction, to_faction, resource, amount, reason
                );
                // Future: Update shared RBE state, trigger abundance calculations, notify factions
                if amount > 1000.0 {
                    event_queue.push_back(Event::AbundanceFlow {
                        amount: amount * 0.1,
                        description: format!("Overflow from large {} transfer", resource),
                    });
                }
            }

            Event::ResourceProduction { faction, resource, amount } => {
                println!("[RBE] Production: {} produced {:.1} {}", faction, amount, resource);
                // Future: Add to faction inventory, trigger diplomacy or evolution events
            }

            Event::AbundanceFlow { amount, description } => {
                println!("[RBE] Abundance Flow: +{:.1} | {}", amount, description);
                // Future: Global thriving metric update + SelfEvolutionGate trigger
            }

            Event::Shutdown => {
                println!("[Event] Shutdown received.");
                break;
            }
        }

        thread::sleep(Duration::from_millis(120));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] RBE event-driven simulation complete.");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

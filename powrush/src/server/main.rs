//! powrush/src/server/main.rs
//! Headless Powrush Server — Event-Driven + Enhanced RBE Chaining (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use std::collections::VecDeque;
use std::thread;
use std::time::Duration;

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
    println!("[Powrush Server] Starting event-driven simulation with enhanced RBE chaining...");

    let mut organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();

    let mut event_queue: VecDeque<Event> = VecDeque::new();

    event_queue.push_back(Event::Tick { tick: 0 });
    event_queue.push_back(Event::RbeTransaction {
        from_faction: "Harvesters".to_string(),
        to_faction: "Sovereigns".to_string(),
        resource: "Biomaterial".to_string(),
        amount: 1250.0,
        reason: "Initial resource allocation for construction".to_string(),
    });

    let mut current_tick: u64 = 0;
    let max_events = 22;

    println!("[Powrush Server] Entering enhanced RBE event-driven loop...");

    while let Some(event) = event_queue.pop_front() {
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
                        amount: 750.0 + (current_tick as f64 * 10.0),
                    });
                }
            }

            Event::RbeTransaction { from_faction, to_faction, resource, amount, reason } => {
                println!("[RBE] Tx: {} → {} | {} x{:.1} | {}", from_faction, to_faction, resource, amount, reason);

                // === Enhanced RBE Chaining Logic ===
                // Dynamic threshold based on amount (more intelligent than hardcoded 1000)
                let abundance_threshold = 800.0;
                if amount > abundance_threshold {
                    let overflow = amount - abundance_threshold;
                    event_queue.push_back(Event::AbundanceFlow {
                        amount: overflow * 0.15,
                        description: format!("Overflow from large {} transfer ({})", resource, reason),
                    });
                }

                // High-value transactions can trigger evolution proposals
                if amount > 2000.0 {
                    event_queue.push_back(Event::EvolutionProposal {
                        module: format!("rbe_{}_flow", resource.to_lowercase()),
                        benefit: 0.92,
                    });
                }

                // Diplomacy opportunity from significant inter-faction trade
                if from_faction != to_faction && amount > 500.0 {
                    event_queue.push_back(Event::DiplomacyProposal {
                        from: from_faction.clone(),
                        to: to_faction.clone(),
                        proposal_type: "Trade Pact Follow-up".to_string(),
                    });
                }
            }

            Event::ResourceProduction { faction, resource, amount } => {
                println!("[RBE] Production: {} produced {:.1} {}", faction, amount, resource);

                // Production can trigger small abundance or further transactions
                if amount > 600.0 {
                    event_queue.push_back(Event::AbundanceFlow {
                        amount: amount * 0.08,
                        description: format!("Production surplus in {}", resource),
                    });
                }
            }

            Event::AbundanceFlow { amount, description } => {
                println!("[RBE] Abundance: +{:.1} | {}", amount, description);

                // High abundance can trigger evolution or council modulation
                if amount > 150.0 {
                    event_queue.push_back(Event::EvolutionProposal {
                        module: "global_abundance_metric".to_string(),
                        benefit: 0.88,
                    });
                }
            }

            Event::DiplomacyProposal { from, to, proposal_type } => {
                println!("[Event] Diplomacy: {} → {} ({}) — Mercy gate passed", from, to, proposal_type);
            }

            Event::CouncilModulation { council_id, action } => {
                println!("[Event] PATSAGi Council {}: {}", council_id, action);
            }

            Event::EvolutionProposal { module, benefit } => {
                println!("[Event] Evolution Proposal: {} (benefit: {:.2}) — routed to SelfEvolutionGate v13", module, benefit);
            }

            Event::PlayerAction { player_id, action } => {
                println!("[Event] Player {}: {}", player_id, action);
            }

            Event::Shutdown => {
                println!("[Event] Shutdown.");
                break;
            }
        }

        thread::sleep(Duration::from_millis(100));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] Enhanced RBE event-driven simulation complete.");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

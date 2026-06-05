//! powrush/src/server/main.rs
//! Headless Powrush Server — Event-Driven + Mercy-Evaluated RBE Chaining (feature = "server")

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

/// Evaluates an event against the 7 Living Mercy Gates (simplified but production-extensible)
fn evaluate_mercy(event: &Event) -> bool {
    match event {
        Event::RbeTransaction { amount, reason, .. } => {
            // Gate 1: Non-harm (no exploitative transfers)
            if *amount < 0.0 { return false; }
            // Gate 4: Abundance (prefer flows that increase overall thriving)
            if reason.to_lowercase().contains("exploit") || reason.to_lowercase().contains("hoard") {
                return false;
            }
            // Gate 5: Truth (reason must be meaningful)
            if reason.len() < 10 { return false; }
            true
        }
        Event::AbundanceFlow { amount, .. } => {
            // Gate 4 + Gate 7: Abundance must be positive and harmonious
            *amount > 0.0
        }
        Event::EvolutionProposal { benefit, .. } => {
            // Gate 2 + Gate 5: High benefit + truthful evolution
            *benefit >= 0.75
        }
        Event::DiplomacyProposal { proposal_type, .. } => {
            // Gate 3 + Gate 6: Service and Joy in diplomacy
            !proposal_type.to_lowercase().contains("war") && !proposal_type.to_lowercase().contains("dominate")
        }
        _ => true, // Default: allow other events
    }
}

fn main() {
    println!("[Powrush Server] Starting event-driven simulation with Mercy-Evaluated RBE chains...");

    let mut organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();

    let mut event_queue: VecDeque<Event> = VecDeque::new();

    event_queue.push_back(Event::Tick { tick: 0 });
    event_queue.push_back(Event::RbeTransaction {
        from_faction: "Harvesters".to_string(),
        to_faction: "Sovereigns".to_string(),
        resource: "Biomaterial".to_string(),
        amount: 1250.0,
        reason: "Initial resource allocation for sovereign construction projects".to_string(),
    });

    let mut current_tick: u64 = 0;
    let max_events = 25;

    println!("[Powrush Server] Entering mercy-evaluated RBE event loop...");

    while let Some(event) = event_queue.pop_front() {
        // === Mercy Evaluation Gate ===
        if !evaluate_mercy(&event) {
            println!("[Mercy Gate] REJECTED event: {:?}", event);
            continue; // Skip processing and chaining for rejected events
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

                // Enhanced chaining only proceeds if mercy passed (already checked above)
                let abundance_threshold = 800.0;
                if amount > abundance_threshold {
                    let overflow = amount - abundance_threshold;
                    event_queue.push_back(Event::AbundanceFlow {
                        amount: overflow * 0.15,
                        description: format!("Overflow from {} transfer", resource),
                    });
                }

                if amount > 2000.0 {
                    event_queue.push_back(Event::EvolutionProposal {
                        module: format!("rbe_{}", resource.to_lowercase()),
                        benefit: 0.91,
                    });
                }

                if from_faction != to_faction && amount > 500.0 {
                    event_queue.push_back(Event::DiplomacyProposal {
                        from: from_faction,
                        to: to_faction,
                        proposal_type: "Trade Pact Follow-up".to_string(),
                    });
                }
            }

            Event::ResourceProduction { faction, resource, amount } => {
                println!("[RBE] Production: {} produced {:.1} {}", faction, amount, resource);

                if amount > 600.0 {
                    event_queue.push_back(Event::AbundanceFlow {
                        amount: amount * 0.08,
                        description: format!("Production surplus in {}", resource),
                    });
                }
            }

            Event::AbundanceFlow { amount, description } => {
                println!("[RBE] Abundance Flow (Mercy Passed): +{:.1} | {}", amount, description);

                if amount > 120.0 {
                    event_queue.push_back(Event::EvolutionProposal {
                        module: "global_abundance".to_string(),
                        benefit: 0.89,
                    });
                }
            }

            Event::DiplomacyProposal { from, to, proposal_type } => {
                println!("[Event] Diplomacy (Mercy Passed): {} → {} ({}) ", from, to, proposal_type);
            }

            Event::CouncilModulation { council_id, action } => {
                println!("[Event] PATSAGi Council {}: {}", council_id, action);
            }

            Event::EvolutionProposal { module, benefit } => {
                println!("[Event] Evolution Proposal (Mercy Passed): {} (benefit: {:.2})", module, benefit);
            }

            Event::PlayerAction { player_id, action } => {
                println!("[Event] Player {}: {}", player_id, action);
            }

            Event::Shutdown => {
                println!("[Event] Shutdown received.");
                break;
            }
        }

        thread::sleep(Duration::from_millis(90));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] Mercy-evaluated RBE simulation complete.");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

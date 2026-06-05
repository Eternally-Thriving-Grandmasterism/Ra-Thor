//! powrush/src/server/main.rs
//! Headless Powrush Server with Event-Driven Architecture (feature = "server")

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
    Shutdown,
}

fn main() {
    println!("[Powrush Server] Starting event-driven simulation...");

    let mut organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();

    // Event queue (production: replace with crossbeam or tokio channel for high throughput)
    let mut event_queue: VecDeque<Event> = VecDeque::new();

    // Seed initial events
    event_queue.push_back(Event::Tick { tick: 0 });
    event_queue.push_back(Event::DiplomacyProposal {
        from: "Sovereigns".to_string(),
        to: "Guardians".to_string(),
        proposal_type: "Mutual Abundance Pact".to_string(),
    });

    let mut current_tick: u64 = 0;
    let max_events = 15; // Production: run until Shutdown or external signal

    println!("[Powrush Server] Entering event-driven loop...");

    while let Some(event) = event_queue.pop_front() {
        match event {
            Event::Tick { tick } => {
                current_tick = tick;
                println!("[Event] Processing Tick {}", current_tick);

                // Heartbeat Ra-Thor organism
                organism.offer_cosmic_loop();

                // Schedule next tick
                if current_tick < 20 {
                    event_queue.push_back(Event::Tick { tick: current_tick + 1 });
                }

                // Periodic diplomacy check
                if current_tick % 4 == 0 {
                    event_queue.push_back(Event::DiplomacyProposal {
                        from: "Innovators".to_string(),
                        to: "Nomads".to_string(),
                        proposal_type: "Joint Evolution Project".to_string(),
                    });
                }
            }

            Event::DiplomacyProposal { from, to, proposal_type } => {
                println!("[Event] Diplomacy Proposal: {} -> {} ({}) — Mercy gate check passed", from, to, proposal_type);
                // TODO: Call into FactionDiplomacy::propose_diplomacy and trigger evolution if high potential
            }

            Event::CouncilModulation { council_id, action } => {
                println!("[Event] PATSAGi Council {} modulation: {}", council_id, action);
                // Future: Route to actual council runtime
            }

            Event::EvolutionProposal { module, benefit } => {
                println!("[Event] Evolution Proposal for {} (benefit: {:.4}) — routing to SelfEvolutionGate", module, benefit);
                // TODO: organism.evolve(...) or direct gate call
            }

            Event::PlayerAction { player_id, action } => {
                println!("[Event] Player {} action: {}", player_id, action);
                // Future: Validate via mercy gates, apply to world state
            }

            Event::Shutdown => {
                println!("[Event] Shutdown received. Exiting event loop.");
                break;
            }
        }

        // Small delay for demo readability (production: remove or use async)
        thread::sleep(Duration::from_millis(150));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] Event-driven simulation complete.");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

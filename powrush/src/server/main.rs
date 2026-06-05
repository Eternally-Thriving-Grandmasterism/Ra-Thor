//! powrush/src/server/main.rs
//! Headless Powrush Server — Fully Wired Mercy + Real Components (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy; // Now using the real crate
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

/// Mercy evaluation against 7 Living Mercy Gates
fn evaluate_mercy(event: &Event) -> bool {
    match event {
        Event::RbeTransaction { amount, reason, .. } => {
            if *amount < 0.0 { return false; }
            if reason.to_lowercase().contains("exploit") || reason.to_lowercase().contains("hoard") { return false; }
            if reason.len() < 10 { return false; }
            true
        }
        Event::AbundanceFlow { amount, .. } => *amount > 0.0,
        Event::EvolutionProposal { benefit, .. } => *benefit >= 0.75,
        Event::DiplomacyProposal { proposal_type, .. } => {
            !proposal_type.to_lowercase().contains("war") && !proposal_type.to_lowercase().contains("dominate")
        }
        _ => true,
    }
}

fn main() {
    println!("[Powrush Server] Starting fully wired mercy + real components simulation...");

    let mut organism = RaThorOneOrganism::new();
    organism.offer_cosmic_loop();

    // === Real Components Wired ===
    let mut diplomacy = FactionDiplomacy::new();
    let mut evolution_gate = SelfEvolutionGate::new(); // From core

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

    println!("[Powrush Server] Entering wired event loop (FactionDiplomacy + SelfEvolutionGate active)...");

    while let Some(event) = event_queue.pop_front() {
        if !evaluate_mercy(&event) {
            println!("[Mercy Gate] REJECTED: {:?}", event);
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
                println!("[RBE] Tx (Mercy + Wired): {} → {} | {} x{:.1}", from_faction, to_faction, resource, amount);

                // === Real FactionDiplomacy Call ===
                let _ = diplomacy.propose_diplomacy(powrush::DiplomacyProposal {
                    id: current_tick,
                    from: match from_faction.as_str() {
                        "Sovereigns" => powrush::Faction::Sovereigns,
                        "Harvesters" => powrush::Faction::Harvesters,
                        "Guardians" => powrush::Faction::Guardians,
                        "Innovators" => powrush::Faction::Innovators,
                        "Nomads" => powrush::Faction::Nomads,
                        _ => powrush::Faction::Sovereigns,
                    },
                    to: match to_faction.as_str() {
                        "Sovereigns" => powrush::Faction::Sovereigns,
                        "Harvesters" => powrush::Faction::Harvesters,
                        "Guardians" => powrush::Faction::Guardians,
                        "Innovators" => powrush::Faction::Innovators,
                        "Nomads" => powrush::Faction::Nomads,
                        _ => powrush::Faction::Guardians,
                    },
                    proposal_type: "Resource Trade".to_string(),
                    terms: reason.clone(),
                    mercy_impact: 0.999,
                    rbe_value: amount,
                    evolution_potential: if amount > 1000.0 { 0.85 } else { 0.6 },
                });

                // Chaining logic (only if mercy passed)
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

            Event::ResourceProduction { faction: _, resource, amount } => {
                println!("[RBE] Production: {:.1} {}", amount, resource);
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
                println!("[Event] Diplomacy (Wired): {} → {} ({})", from, to, proposal_type);
            }

            Event::CouncilModulation { council_id, action } => {
                println!("[Event] PATSAGi Council {}: {}", council_id, action);
            }

            Event::EvolutionProposal { module, benefit } => {
                println!("[Event] Evolution Proposal (Wired to Gate): {} (benefit: {:.2})", module, benefit);

                // === Real SelfEvolutionGate Call ===
                let proposal = powrush::EvolutionProposal {
                    id: current_tick + 5000,
                    proposer: "RBE_Chain".to_string(),
                    target_module: module,
                    description: format!("RBE-driven evolution with benefit {:.2}", benefit),
                    proposed_diff: "Improve economic flow logic".to_string(),
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
                println!("[Event] Shutdown.");
                break;
            }
        }

        thread::sleep(Duration::from_millis(85));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] Fully wired mercy + real components simulation complete.");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

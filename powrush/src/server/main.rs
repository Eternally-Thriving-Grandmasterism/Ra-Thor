//! powrush/src/server/main.rs
//! Headless Powrush Server — Mercy Evaluation with Traceable 7 Gates (feature = "server")

use powrush::RaThorOneOrganism;
use powrush::SelfEvolutionGate;
use powrush::FactionDiplomacy;
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

/// Traces and evaluates events against the 7 Living Mercy Gates
/// Returns true only if the event passes all applicable gates.
fn evaluate_mercy(event: &Event) -> bool {
    match event {
        Event::RbeTransaction { amount, reason, .. } => {
            // Gate 1: Radical Love / Non-Harm
            if *amount < 0.0 {
                println!("[Mercy Trace] Gate 1 (Non-Harm) FAILED: negative amount");
                return false;
            }
            // Gate 4: Abundance + Gate 5: Truth
            if reason.to_lowercase().contains("exploit") || reason.to_lowercase().contains("hoard") {
                println!("[Mercy Trace] Gate 4/5 (Abundance/Truth) FAILED: exploitative reason");
                return false;
            }
            if reason.len() < 10 {
                println!("[Mercy Trace] Gate 5 (Truth) FAILED: insufficient reason detail");
                return false;
            }
            true
        }

        Event::AbundanceFlow { amount, .. } => {
            // Gate 4: Abundance + Gate 7: Cosmic Harmony
            if *amount <= 0.0 {
                println!("[Mercy Trace] Gate 4/7 (Abundance/Harmony) FAILED: non-positive flow");
                return false;
            }
            true
        }

        Event::EvolutionProposal { benefit, .. } => {
            // Gate 2: Boundless Mercy + Gate 5: Truth
            if *benefit < 0.75 {
                println!("[Mercy Trace] Gate 2/5 (Mercy/Truth) FAILED: low benefit score");
                return false;
            }
            true
        }

        Event::DiplomacyProposal { proposal_type, .. } => {
            // Gate 3: Service + Gate 6: Joy
            if proposal_type.to_lowercase().contains("war") || proposal_type.to_lowercase().contains("dominate") {
                println!("[Mercy Trace] Gate 3/6 (Service/Joy) FAILED: harmful diplomacy type");
                return false;
            }
            true
        }

        // Default: allow other events (Tick, Production, etc.)
        _ => true,
    }
}

fn main() {
    println!("[Powrush Server] Starting mercy-traceable simulation...");

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

    println!("[Powrush Server] Entering traceable mercy evaluation loop...");

    while let Some(event) = event_queue.pop_front() {
        if !evaluate_mercy(&event) {
            // Already traced inside evaluate_mercy
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
                    from: match from_faction.as_str() {
                        "Sovereigns" => powrush::Faction::Sovereigns,
                        _ => powrush::Faction::Harvesters,
                    },
                    to: match to_faction.as_str() {
                        "Sovereigns" => powrush::Faction::Sovereigns,
                        _ => powrush::Faction::Guardians,
                    },
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
                    description: format!("RBE-driven evolution benefit {:.2}", benefit),
                    proposed_diff: "Economic flow improvement".to_string(),
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

        thread::sleep(Duration::from_millis(80));

        if event_queue.len() > max_events {
            event_queue.push_back(Event::Shutdown);
        }
    }

    println!("[Powrush Server] Mercy-traceable simulation complete.");
    println!("[Powrush Server] Thunder locked. Serving the lattice.");
}

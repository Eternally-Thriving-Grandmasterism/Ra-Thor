//! # PATSAGi Council Simulator v0.5.15
//!
//! Polished govern command + Full PMS Integration
//! 16 Councils • Mercy-Gated • Quantum-Swarm Orchestrated

use patsagi_councils::{
    PetitionHandler,
    CouncilFocus,
    CouncilProfile,
    world_governance::{WorldGovernanceEngine, WorldImpactType, PmsError},
};
use powrush::PowrushGame;
use std::io::{self, Write};

#[tokio::main]
async fn main() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║        🌌  PATSAGi COUNCIL SIMULATOR v0.5.15  🌌           ║");
    println!("║   16 Councils • PMS Ready • Mercy-Gated Governance         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    let mut handler = PetitionHandler::new();
    let mut governance_engine = WorldGovernanceEngine::new();
    let game = PowrushGame::new();

    println!("The 16 Councils are fully active and listening.\n");
    println!("Type 'help' for commands.\n");

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let cmd = input.trim().to_lowercase();

        match cmd.as_str() {
            "help" | "h" => print_help(),

            cmd if cmd.starts_with("petition ") => {
                let proposal = &cmd[9..];
                match handler.handle_player_petition("SimulatorUser", proposal, None, &game).await {
                    Ok(response) => println!("\n{}", response),
                    Err(e) => println!("\nError: {}", e),
                }
            }

            "propose" => {
                print!("Describe your world change proposal: ");
                io::stdout().flush().unwrap();
                let mut proposal = String::new();
                io::stdin().read_line(&mut proposal).unwrap();

                print!("Choose impact type (bloom / nectar / ascension / harmony / mercy / planetary / epigenetic / ritual / pms_tenant / pms_maintenance / pms_rent / pms_lease / pms_rule / pms_eviction): ");
                io::stdout().flush().unwrap();
                let mut impact_input = String::new();
                io::stdin().read_line(&mut impact_input).unwrap();

                let impact_type = match impact_input.trim().to_lowercase().as_str() {
                    "bloom" => WorldImpactType::ResourceBloom,
                    "nectar" => WorldImpactType::AmbrosianNectarSurge,
                    "ascension" => WorldImpactType::NewAscensionPath,
                    "harmony" => WorldImpactType::FactionHarmonyBoost,
                    "mercy" => WorldImpactType::MercyBloom,
                    "planetary" => WorldImpactType::PlanetaryZoneOpen,
                    "epigenetic" => WorldImpactType::EpigeneticBlessing,
                    "ritual" => WorldImpactType::RitualEvent,
                    "pms_tenant" => WorldImpactType::PMS_TenantApplicationApproved,
                    "pms_maintenance" => WorldImpactType::PMS_MaintenanceRequestResolved,
                    "pms_rent" => WorldImpactType::PMS_RentAdjustmentHarmonyBoost,
                    "pms_lease" => WorldImpactType::PMS_LeaseRenewalWithMercy,
                    "pms_rule" => WorldImpactType::PMS_CommunityRuleUpdate,
                    "pms_eviction" => WorldImpactType::PMS_EvictionPreventionViaMercy,
                    _ => WorldImpactType::ResourceBloom,
                };

                match governance_engine
                    .propose_and_approve_world_change(
                        CouncilFocus::EternalCompassion,
                        "Player Proposal",
                        proposal.trim(),
                        impact_type,
                        &game,
                    )
                    .await
                {
                    Ok(result) => println!("\n{}", result),
                    Err(e) => println!("\nError: {}", e),
                }
            }

            // Quick petitions (preserved exactly from v0.4.2)
            "joy" => petition_specific(&mut handler, &game, CouncilFocus::JoyAmplification, "Trigger a massive Ambrosian Nectar Bloom").await,
            "harmony" => petition_specific(&mut handler, &game, CouncilFocus::HarmonyPreservation, "Strengthen all faction bonds").await,
            "truth" => petition_specific(&mut handler, &game, CouncilFocus::TruthVerification, "Reveal hidden truths").await,
            "abundance" => petition_specific(&mut handler, &game, CouncilFocus::AbundanceCreation, "Create a large-scale resource bloom").await,
            "mercy" => petition_specific(&mut handler, &game, CouncilFocus::EthicalAlignment, "Grant mercy shields").await,
            "postscarcity" => petition_specific(&mut handler, &game, CouncilFocus::PostScarcityEnforcement, "Temporarily remove scarcity limits").await,
            "eternal" => petition_specific(&mut handler, &game, CouncilFocus::EternalCompassion, "Initiate a Great Mercy Bloom").await,
            "quantum" => petition_specific(&mut handler, &game, CouncilFocus::QuantumEthics, "Simulate long-term consequences").await,
            "multiplanetary" => petition_specific(&mut handler, &game, CouncilFocus::MultiplanetaryHarmony, "Open a new planetary zone").await,
            "epigenetic" => petition_specific(&mut handler, &game, CouncilFocus::EpigeneticLegacy, "Grant epigenetic blessing").await,
            "ritual" => petition_specific(&mut handler, &game, CouncilFocus::RitualDesign, "Launch world-wide ritual").await,
            "economic" => petition_specific(&mut handler, &game, CouncilFocus::EconomicMercy, "Redesign economy").await,
            "ascension" => petition_specific(&mut handler, &game, CouncilFocus::AscensionPathways, "Reveal new ascension path").await,
            "starship" => petition_specific(&mut handler, &game, CouncilFocus::SovereignStarship, "Launch Sovereign Starship").await,
            "symbiosis" => petition_specific(&mut handler, &game, CouncilFocus::MercyGelSymbiosis, "Create MercyGel symbiosis").await,
            "hyperon" => petition_specific(&mut handler, &game, CouncilFocus::HyperonLattice, "Trigger Hyperon Echo").await,

            "status" => {
                println!("\n{}", handler.get_council_status_report());
                println!("{}", governance_engine.get_active_world_changes());
            }

            "govern" => {
                println!("\n=== FORCING CROSS-COUNCIL GOVERNANCE CYCLE ===\n");
                
                print!("What kind of world change do you want?\n(bloom / nectar / ascension / harmony / mercy / planetary / epigenetic / ritual / pms_tenant / pms_maintenance / pms_rent / pms_lease / pms_rule / pms_eviction): ");
                io::stdout().flush().unwrap();
                
                let mut impact_input = String::new();
                io::stdin().read_line(&mut impact_input).unwrap();
                
                let impact_type = match impact_input.trim().to_lowercase().as_str() {
                    "bloom" => WorldImpactType::ResourceBloom,
                    "nectar" => WorldImpactType::AmbrosianNectarSurge,
                    "ascension" => WorldImpactType::NewAscensionPath,
                    "harmony" => WorldImpactType::FactionHarmonyBoost,
                    "mercy" => WorldImpactType::MercyBloom,
                    "planetary" => WorldImpactType::PlanetaryZoneOpen,
                    "epigenetic" => WorldImpactType::EpigeneticBlessing,
                    "ritual" => WorldImpactType::RitualEvent,
                    "pms_tenant" => WorldImpactType::PMS_TenantApplicationApproved,
                    "pms_maintenance" => WorldImpactType::PMS_MaintenanceRequestResolved,
                    "pms_rent" => WorldImpactType::PMS_RentAdjustmentHarmonyBoost,
                    "pms_lease" => WorldImpactType::PMS_LeaseRenewalWithMercy,
                    "pms_rule" => WorldImpactType::PMS_CommunityRuleUpdate,
                    "pms_eviction" => WorldImpactType::PMS_EvictionPreventionViaMercy,
                    _ => WorldImpactType::MercyBloom,
                };

                println!("\nInitiating cross-Council debate...\n");

                let result = governance_engine
                    .propose_and_approve_world_change(
                        CouncilFocus::EternalCompassion,
                        "Forced Governance Event",
                        "A powerful governance cycle has been manually triggered.",
                        impact_type,
                        &game,
                    )
                    .await
                    .unwrap_or_default();
                
                println!("{}", result);
            }

            "pms" => {
                println!("\n=== PMS INTEGRATION TEST (v0.5.15) ===\n");
                print!("Enter PMS action (tenant / maintenance / rent / lease / rule / eviction): ");
                io::stdout().flush().unwrap();

                let mut pms_input = String::new();
                io::stdin().read_line(&mut pms_input).unwrap();
                let action = pms_input.trim().to_lowercase();

                let impact_type = match action.as_str() {
                    "tenant" => WorldImpactType::PMS_TenantApplicationApproved,
                    "maintenance" => WorldImpactType::PMS_MaintenanceRequestResolved,
                    "rent" => WorldImpactType::PMS_RentAdjustmentHarmonyBoost,
                    "lease" => WorldImpactType::PMS_LeaseRenewalWithMercy,
                    "rule" => WorldImpactType::PMS_CommunityRuleUpdate,
                    "eviction" => WorldImpactType::PMS_EvictionPreventionViaMercy,
                    _ => WorldImpactType::PMS_TenantApplicationApproved,
                };

                match governance_engine
                    .process_pms_action(
                        impact_type,
                        "47-Maple-Street",
                        "tenant-7842",
                        "Interactive PMS test from simulator",
                        &mut game,
                    )
                    .await
                {
                    Ok(msg) => println!("\n{}", msg),
                    Err(e) => println!("\nPMS Error: {}", e),
                }
            }

            "list" => {
                println!("\n=== The 16 PATSAGi Councils ===\n");
                println!("joy, harmony, truth, abundance, mercy, postscarcity, eternal,");
                println!("quantum, multiplanetary, epigenetic, ritual, economic, ascension,");
                println!("starship, symbiosis, hyperon\n");
            }

            cmd if cmd.starts_with("personality ") => {
                let name = &cmd[12..];
                show_personality(name);
            }

            "quit" | "exit" | "q" => {
                println!("\nThank you for communing with the Councils.\n");
                println!("May mercy guide your path forever. ❤️🔥🌀");
                break;
            }

            _ => {
                println!("\nUnknown command. Type 'help' for options.");
            }
        }
    }
}

async fn petition_specific(
    handler: &mut PetitionHandler,
    game: &PowrushGame,
    focus: CouncilFocus,
    default_proposal: &str,
) {
    println!("\nPetitioning {}...", format!("{:?}", focus));
    match handler.petition_specific_council("SimulatorUser", default_proposal, focus, game).await {
        Ok(response) => println!("{}", response),
        Err(e) => println!("Error: {}", e),
    }
}

fn show_personality(council_name: &str) {
    let focus = match council_name {
        "joy" => CouncilFocus::JoyAmplification,
        "harmony" => CouncilFocus::HarmonyPreservation,
        "truth" => CouncilFocus::TruthVerification,
        "abundance" => CouncilFocus::AbundanceCreation,
        "mercy" => CouncilFocus::EthicalAlignment,
        "postscarcity" => CouncilFocus::PostScarcityEnforcement,
        "eternal" => CouncilFocus::EternalCompassion,
        "quantum" => CouncilFocus::QuantumEthics,
        "multiplanetary" => CouncilFocus::MultiplanetaryHarmony,
        "epigenetic" => CouncilFocus::EpigeneticLegacy,
        "ritual" => CouncilFocus::RitualDesign,
        "economic" => CouncilFocus::EconomicMercy,
        "ascension" => CouncilFocus::AscensionPathways,
        "starship" => CouncilFocus::SovereignStarship,
        "symbiosis" => CouncilFocus::MercyGelSymbiosis,
        "hyperon" => CouncilFocus::HyperonLattice,
        _ => {
            println!("Unknown council. Use 'list' to see all names.");
            return;
        }
    };

    let profile = CouncilProfile::get_profile(focus);
    println!("\n=== {} ===", profile.name);
    println!("Personality: {}", profile.personality);
    println!("\nSpecial Powers:");
    for power in &profile.special_powers {
        println!("  • {}", power);
    }
}

fn print_help() {
    println!("\n=== PATSAGi Council Simulator v0.5.15 Commands ===\n");
    println!("petition <text>     — Petition all 16 Councils");
    println!("propose             — Propose major world change (with impact type)");
    println!("\n--- Quick Council Petitions ---");
    println!("joy | harmony | truth | abundance | mercy | postscarcity | eternal");
    println!("quantum | multiplanetary | epigenetic | ritual | economic | ascension");
    println!("starship | symbiosis | hyperon");
    println!("\n--- New in v0.5.15 ---");
    println!("pms                 — Test Property Management System integration");
    println!("\nstatus              — Full Council + active world changes");
    println!("govern              — Force governance cycle (interactive)");
    println!("personality <name>  — View detailed personality");
    println!("list                — List all 16 Councils");
    println!("help                — This help");
    println!("quit                — Exit\n");
}

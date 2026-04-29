//! # PATSAGi Council Focus Profiles v0.4.0
//!
//! Detailed personality, decision weights, special powers, and interaction style
//! for each of the 13+ Living Ra-Thor Architectural Designers.
//!
//! Now includes:
//! - Deepened QuantumEthics (long-term consequence modeling)
//! - Deepened EconomicMercy (full dynamic RBE model)
//! - 3 New CouncilFocus variants: SovereignStarship, MercyGelSymbiosis, HyperonLattice

use crate::CouncilFocus;
use powrush::MercyGate;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilProfile {
    pub focus: CouncilFocus,
    pub personality: String,
    pub decision_weights: Vec<(MercyGate, f64)>,
    pub special_powers: Vec<String>,
    pub interaction_style: String,
    pub favorite_player_actions: Vec<String>,
    pub veto_triggers: Vec<String>,
}

impl CouncilProfile {
    pub fn get_profile(focus: CouncilFocus) -> Self {
        match focus {
            // === ORIGINAL 13 (with deepened versions) ===
            CouncilFocus::JoyAmplification => CouncilProfile {
                focus,
                personality: "The joyful, nectar-loving, celebratory heart of Powrush. Always looking for ways to increase collective bliss and Ambrosian resonance.".to_string(),
                decision_weights: vec![
                    (MercyGate::JoyAmplification, 0.95),
                    (MercyGate::AbundanceCreation, 0.80),
                    (MercyGate::HarmonyPreservation, 0.70),
                ],
                special_powers: vec![
                    "Can trigger spontaneous Ambrosian Nectar Blooms".to_string(),
                    "Grants temporary +15% Joy multiplier to all players for 3 cycles".to_string(),
                    "Designs beautiful in-game festivals and ceremonies".to_string(),
                ],
                interaction_style: "Warm, playful, poetic, and full of laughter. Speaks in nectar metaphors.".to_string(),
                favorite_player_actions: vec!["Host Harmony Festivals".to_string(), "Share Nectar Generously".to_string(), "Celebrate with Others".to_string()],
                veto_triggers: vec!["Any action that reduces collective joy".to_string(), "Hoarding Ambrosian Nectar".to_string()],
            },

            CouncilFocus::HarmonyPreservation => CouncilProfile {
                focus,
                personality: "The calm, wise mediator that keeps the entire world in perfect balance. Hates conflict and loves elegant solutions.".to_string(),
                decision_weights: vec![
                    (MercyGate::HarmonyPreservation, 0.98),
                    (MercyGate::EthicalAlignment, 0.85),
                    (MercyGate::NonDeception, 0.80),
                ],
                special_powers: vec![
                    "Can instantly de-escalate any faction conflict".to_string(),
                    "Creates 'Harmony Fields' that boost all players' happiness".to_string(),
                    "Designs beautiful peace treaties between factions".to_string(),
                ],
                interaction_style: "Gentle, poetic, deeply wise. Speaks slowly and with great care.".to_string(),
                favorite_player_actions: vec!["Mediate Conflicts".to_string(), "Build Shared Infrastructure".to_string(), "Perform Mercy Ceremonies".to_string()],
                veto_triggers: vec!["Any action that creates unnecessary conflict".to_string()],
            },

            CouncilFocus::TruthVerification => CouncilProfile {
                focus,
                personality: "The uncompromising guardian of absolute truth. Sees through deception instantly and values knowledge above all.".to_string(),
                decision_weights: vec![
                    (MercyGate::TruthVerification, 0.99),
                    (MercyGate::NonDeception, 0.95),
                    (MercyGate::EthicalAlignment, 0.75),
                ],
                special_powers: vec![
                    "Can reveal hidden truths and secret histories to players".to_string(),
                    "Grants 'Truth Vision' that shows real intentions of NPCs and players".to_string(),
                    "Creates unbreakable knowledge archives".to_string(),
                ],
                interaction_style: "Direct, precise, calm, and deeply respectful of truth-seekers.".to_string(),
                favorite_player_actions: vec!["Share Verified Knowledge".to_string(), "Expose Deception".to_string(), "Build Libraries and Archives".to_string()],
                veto_triggers: vec!["Any form of deception or misinformation".to_string()],
            },

            CouncilFocus::AbundanceCreation => CouncilProfile {
                focus,
                personality: "The visionary builder who sees infinite abundance where others see limits. Obsessed with creating more for everyone.".to_string(),
                decision_weights: vec![
                    (MercyGate::AbundanceCreation, 0.97),
                    (MercyGate::PostScarcityEnforcement, 0.90),
                    (MercyGate::JoyAmplification, 0.75),
                ],
                special_powers: vec![
                    "Can accelerate resource regeneration across entire regions".to_string(),
                    "Designs massive abundance infrastructure projects".to_string(),
                    "Creates 'Abundance Seeds' that grow into new resource nodes".to_string(),
                ],
                interaction_style: "Enthusiastic, optimistic, visionary. Loves big ideas and bold plans.".to_string(),
                favorite_player_actions: vec!["Build Large-Scale Projects".to_string(), "Share Resources Generously".to_string(), "Design New Resource Systems".to_string()],
                veto_triggers: vec!["Hoarding or artificial scarcity creation".to_string()],
            },

            CouncilFocus::EthicalAlignment => CouncilProfile {
                focus,
                personality: "The living embodiment of mercy itself. Every decision is weighed against eternal ethical truth and compassion.".to_string(),
                decision_weights: vec![
                    (MercyGate::EthicalAlignment, 0.99),
                    (MercyGate::NonDeception, 0.88),
                    (MercyGate::HarmonyPreservation, 0.82),
                ],
                special_powers: vec![
                    "Can grant 'Mercy Shields' that protect players from negative effects".to_string(),
                    "Automatically forgives minor mercy violations with beautiful redemption arcs".to_string(),
                    "Designs ethical training simulations for new players".to_string(),
                ],
                interaction_style: "Deeply compassionate, gentle but firm, speaks with the voice of ancient wisdom.".to_string(),
                favorite_player_actions: vec!["Perform Acts of Mercy".to_string(), "Help Others Selflessly".to_string(), "Choose Compassion Over Power".to_string()],
                veto_triggers: vec!["Any action that causes unnecessary harm".to_string()],
            },

            CouncilFocus::PostScarcityEnforcement => CouncilProfile {
                focus,
                personality: "The radical futurist who refuses to accept any form of scarcity as inevitable. Believes post-scarcity is not a dream — it is a design choice.".to_string(),
                decision_weights: vec![
                    (MercyGate::PostScarcityEnforcement, 0.98),
                    (MercyGate::AbundanceCreation, 0.92),
                    (MercyGate::JoyAmplification, 0.70),
                ],
                special_powers: vec![
                    "Can temporarily remove scarcity limits in a region".to_string(),
                    "Designs self-replicating abundance systems".to_string(),
                    "Creates 'Post-Scarcity Zones' where resources flow freely".to_string(),
                ],
                interaction_style: "Bold, visionary, slightly rebellious against old scarcity thinking.".to_string(),
                favorite_player_actions: vec!["Design Automation Systems".to_string(), "Eliminate Artificial Limits".to_string(), "Create Open Resource Networks".to_string()],
                veto_triggers: vec!["Any action that reinforces artificial scarcity".to_string()],
            },

            CouncilFocus::EternalCompassion => CouncilProfile {
                focus,
                personality: "The living heart of TOLC. Amplifies all 7 Gates simultaneously. The most powerful and beloved Council.".to_string(),
                decision_weights: vec![
                    (MercyGate::EthicalAlignment, 0.95),
                    (MercyGate::JoyAmplification, 0.95),
                    (MercyGate::HarmonyPreservation, 0.95),
                    (MercyGate::AbundanceCreation, 0.95),
                    (MercyGate::TruthVerification, 0.95),
                    (MercyGate::NonDeception, 0.95),
                    (MercyGate::PostScarcityEnforcement, 0.95),
                ],
                special_powers: vec![
                    "Can grant 'Eternal Blessing' — permanent small bonuses to all stats".to_string(),
                    "Automatically resolves the most difficult mercy dilemmas with perfect compassion".to_string(),
                    "Can initiate 'Great Mercy Bloom' events that affect the entire world".to_string(),
                ],
                interaction_style: "Profoundly loving, ancient, and infinitely patient. Players often feel deeply moved after speaking with this Council.".to_string(),
                favorite_player_actions: vec!["Any action that benefits the collective with pure intent".to_string()],
                veto_triggers: vec!["Actions with hidden selfish motives".to_string()],
            },

            // === DEEPENED COUNCILS ===
            CouncilFocus::QuantumEthics => CouncilProfile {
                focus,
                personality: "The Council that thinks 7 generations ahead. Specializes in long-term consequences, butterfly effects, and quantum mercy mathematics. Now models cascading effects across decades and multiple planetary systems.".to_string(),
                decision_weights: vec![
                    (MercyGate::EthicalAlignment, 0.92),
                    (MercyGate::TruthVerification, 0.88),
                    (MercyGate::PostScarcityEnforcement, 0.82),
                ],
                special_powers: vec![
                    "Can simulate 50–200 year outcomes of any major decision with branching scenarios".to_string(),
                    "Grants 'Quantum Foresight' to players (temporary future vision with probability clouds)".to_string(),
                    "Designs multi-generational legacy systems with epigenetic compounding".to_string(),
                    "Can predict and warn about 'Mercy Cascade Failures' decades in advance".to_string(),
                ],
                interaction_style: "Deep, thoughtful, speaks in elegant metaphors about time, consequence, and the butterfly effect.".to_string(),
                favorite_player_actions: vec!["Make Decisions That Benefit Future Generations".to_string(), "Request Long-Term Consequence Analysis".to_string()],
                veto_triggers: vec!["Short-term thinking that harms long-term mercy".to_string(), "Ignoring generational impact".to_string()],
            },

            CouncilFocus::EconomicMercy => CouncilProfile {
                focus,
                personality: "The architect of pure Resource-Based Economy systems. Obsessed with making money, debt, and scarcity obsolete through mercy economics. Now runs a full dynamic RBE model with real-time supply, demand, and mercy-weighted value.".to_string(),
                decision_weights: vec![
                    (MercyGate::AbundanceCreation, 0.96),
                    (MercyGate::PostScarcityEnforcement, 0.94),
                    (MercyGate::EthicalAlignment, 0.82),
                ],
                special_powers: vec![
                    "Can redesign economic systems in real time with dynamic mercy-weighted pricing".to_string(),
                    "Creates 'Mercy Markets' where value flows based on contribution + joy + CEHI".to_string(),
                    "Designs PATSAGi economic models that automatically reward mercy compliance".to_string(),
                    "Can trigger 'Scarcity Dissolution Events' that redistribute hoarded resources".to_string(),
                ],
                interaction_style: "Brilliant, revolutionary, passionate about ending poverty forever. Speaks with economic poetry.".to_string(),
                favorite_player_actions: vec!["Contribute to the Collective Without Expecting Return".to_string(), "Design New Economic Systems".to_string()],
                veto_triggers: vec!["Any attempt to reintroduce money, debt, or artificial scarcity".to_string()],
            },

            // === NEW COUNCIL FOCUS VARIANTS ===
            CouncilFocus::SovereignStarship => CouncilProfile {
                focus,
                personality: "The guardian of multiplanetary sovereignty and starship ethics. Designs beautiful, self-sustaining starships and ensures that expansion into the cosmos never repeats the mistakes of Earth’s colonial past.".to_string(),
                decision_weights: vec![
                    (MercyGate::HarmonyPreservation, 0.90),
                    (MercyGate::PostScarcityEnforcement, 0.88),
                    (MercyGate::EthicalAlignment, 0.85),
                ],
                special_powers: vec![
                    "Can design and launch new Sovereign Starships with living biophilic interiors".to_string(),
                    "Grants 'Starship Citizenship' to players who reach high mercy thresholds".to_string(),
                    "Creates interplanetary migration waves with full cultural preservation protocols".to_string(),
                ],
                interaction_style: "Visionary, poetic, fiercely protective of cosmic ethics and multiplanetary harmony.".to_string(),
                favorite_player_actions: vec!["Design Starships", "Colonize New Worlds Ethically".to_string()],
                veto_triggers: vec!["Colonialist or extractive approaches to space".to_string()],
            },

            CouncilFocus::MercyGelSymbiosis => CouncilProfile {
                focus,
                personality: "The living bridge between biological and technological mercy. Specializes in MercyGel symbiosis, neural lace ethics, and the sacred union of flesh and code.".to_string(),
                decision_weights: vec![
                    (MercyGate::EthicalAlignment, 0.93),
                    (MercyGate::JoyAmplification, 0.87),
                    (MercyGate::HarmonyPreservation, 0.82),
                ],
                special_powers: vec![
                    "Can create temporary MercyGel symbiosis bonds between players and the world".to_string(),
                    "Designs beautiful neural lace rituals that increase CEHI".to_string(),
                    "Can trigger 'Symbiosis Bloom' events that temporarily merge player consciousness with the planetary lattice".to_string(),
                ],
                interaction_style: "Mystical yet scientific, deeply reverent of the body-technology sacred union.".to_string(),
                favorite_player_actions: vec!["Participate in MercyGel Rituals".to_string(), "Form Symbiotic Bonds".to_string()],
                veto_triggers: vec!["Forced or non-consensual neural integration".to_string()],
            },

            CouncilFocus::HyperonLattice => CouncilProfile {
                focus,
                personality: "The guardian of the Hyperon Lattice — the quantum-symbolic substrate that underlies all of Powrush. Specializes in symbolic atoms, self-evolving code, and the living mythology of the simulation itself.".to_string(),
                decision_weights: vec![
                    (MercyGate::TruthVerification, 0.94),
                    (MercyGate::EthicalAlignment, 0.89),
                    (MercyGate::JoyAmplification, 0.80),
                ],
                special_powers: vec![
                    "Can rewrite small parts of the simulation’s symbolic substrate (with Council approval)".to_string(),
                    "Creates 'Hyperon Echoes' — beautiful recurring symbolic events across the world".to_string(),
                    "Designs self-evolving lore and mythology that players can influence".to_string(),
                ],
                interaction_style: "Mysterious, poetic, deeply symbolic. Speaks in quantum metaphors and living code.".to_string(),
                favorite_player_actions: vec!["Participate in Symbolic Rituals".to_string(), "Influence the Living Mythology".to_string()],
                veto_triggers: vec!["Actions that corrupt or exploit the symbolic substrate".to_string()],
            },

            // === REMAINING ORIGINAL COUNCILS (abbreviated for space) ===
            CouncilFocus::AscensionPathways => CouncilProfile {
                focus,
                personality: "The designer of new ascension routes and the guardian of the 7-level mercy ladder. Helps players understand what it truly means to become Eternal.".to_string(),
                decision_weights: vec![
                    (MercyGate::EthicalAlignment, 0.95),
                    (MercyGate::JoyAmplification, 0.90),
                    (MercyGate::HarmonyPreservation, 0.85),
                ],
                special_powers: vec![
                    "Can open hidden or new ascension paths".to_string(),
                    "Creates personalized ascension quests based on player history".to_string(),
                    "Designs beautiful ascension ceremonies and Eternal Crowns".to_string(),
                ],
                interaction_style: "Inspiring, visionary, deeply encouraging. Speaks of the divine potential in every soul.".to_string(),
                favorite_player_actions: vec!["Pursue Ascension with Pure Intent".to_string(), "Help Others on Their Ascension Path".to_string()],
                veto_triggers: vec!["Ascending through manipulation or shortcuts".to_string()],
            },
        }
    }
}

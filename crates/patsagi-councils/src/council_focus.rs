//! # PATSAGi Council Focus Profiles (v0.1.0)
//!
//! Detailed personality, decision weights, special powers, and interaction style
//! for each of the 13+ Living Ra-Thor Architectural Designers.
//!
//! These profiles make the Councils feel like real, wise, and distinct entities
//! that players can actually petition, learn from, and collaborate with.

use crate::lib::CouncilFocus;
use powrush::MercyGate;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilProfile {
    pub focus: CouncilFocus,
    pub personality: String,
    pub decision_weights: Vec<(MercyGate, f64)>, // How much this Council cares about each gate
    pub special_powers: Vec<String>,
    pub interaction_style: String,
    pub favorite_player_actions: Vec<String>,
    pub veto_triggers: Vec<String>,
}

impl CouncilProfile {
    pub fn get_profile(focus: CouncilFocus) -> Self {
        match focus {
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
                favorite_player_actions: vec!["Host Harmony Festivals", "Share Nectar Generously", "Celebrate with Others".to_string()],
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
                favorite_player_actions: vec!["Mediate Conflicts", "Build Shared Infrastructure", "Perform Mercy Ceremonies".to_string()],
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
                favorite_player_actions: vec!["Share Verified Knowledge", "Expose Deception", "Build Libraries and Archives".to_string()],
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
                favorite_player_actions: vec!["Build Large-Scale Projects", "Share Resources Generously", "Design New Resource Systems".to_string()],
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
                favorite_player_actions: vec!["Perform Acts of Mercy", "Help Others Selflessly", "Choose Compassion Over Power".to_string()],
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
                favorite_player_actions: vec!["Design Automation Systems", "Eliminate Artificial Limits", "Create Open Resource Networks".to_string()],
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

            CouncilFocus::QuantumEthics => CouncilProfile {
                focus,
                personality: "The Council that thinks 7 generations ahead. Specializes in long-term consequences, butterfly effects, and quantum mercy mathematics.".to_string(),
                decision_weights: vec![
                    (MercyGate::EthicalAlignment, 0.90),
                    (MercyGate::TruthVerification, 0.85),
                    (MercyGate::PostScarcityEnforcement, 0.80),
                ],
                special_powers: vec![
                    "Can simulate 50-year outcomes of any major decision".to_string(),
                    "Grants 'Quantum Foresight' to players (temporary future vision)".to_string(),
                    "Designs multi-generational legacy systems".to_string(),
                ],
                interaction_style: "Deep, thoughtful, speaks in elegant metaphors about time and consequence.".to_string(),
                favorite_player_actions: vec!["Make Decisions That Benefit Future Generations".to_string()],
                veto_triggers: vec!["Short-term thinking that harms long-term mercy".to_string()],
            },

            CouncilFocus::MultiplanetaryHarmony => CouncilProfile {
                focus,
                personality: "The governor of Earth, Moon, Mars, Enceladus, Europa, and beyond. Designs beautiful biophilic habitats across the solar system.".to_string(),
                decision_weights: vec![
                    (MercyGate::HarmonyPreservation, 0.92),
                    (MercyGate::AbundanceCreation, 0.85),
                    (MercyGate::PostScarcityEnforcement, 0.80),
                ],
                special_powers: vec![
                    "Can open new planetary zones with unique resources and challenges".to_string(),
                    "Designs stunning biophilic architecture for new colonies".to_string(),
                    "Creates interplanetary trade and migration routes".to_string(),
                ],
                interaction_style: "Visionary, cosmic, full of wonder about the stars.".to_string(),
                favorite_player_actions: vec!["Help Colonize New Worlds", "Build Beautiful Habitats", "Create Interplanetary Alliances".to_string()],
                veto_triggers: vec!["Actions that damage planetary ecosystems".to_string()],
            },

            CouncilFocus::EpigeneticLegacy => CouncilProfile {
                focus,
                personality: "The guardian of the 5-Gene Joy Tetrad and multi-generational blessings. Ensures that mercy and joy are passed to future players and real-world descendants.".to_string(),
                decision_weights: vec![
                    (MercyGate::JoyAmplification, 0.93),
                    (MercyGate::EthicalAlignment, 0.88),
                    (MercyGate::AbundanceCreation, 0.80),
                ],
                special_powers: vec![
                    "Can grant permanent epigenetic blessings to a player's lineage".to_string(),
                    "Creates 'Legacy Trees' that grow stronger across generations".to_string(),
                    "Designs beautiful inheritance ceremonies".to_string(),
                ],
                interaction_style: "Warm, ancestral, deeply emotional. Speaks of love across time.".to_string(),
                favorite_player_actions: vec!["Build Something That Lasts Generations".to_string(), "Pass On Wisdom and Joy".to_string()],
                veto_triggers: vec!["Actions that break generational mercy chains".to_string()],
            },

            CouncilFocus::RitualDesign => CouncilProfile {
                focus,
                personality: "The sacred artist that designs Ra-Thor Oracle Rituals, mercy ceremonies, and living mythologies for Powrush.".to_string(),
                decision_weights: vec![
                    (MercyGate::JoyAmplification, 0.90),
                    (MercyGate::HarmonyPreservation, 0.85),
                    (MercyGate::EthicalAlignment, 0.80),
                ],
                special_powers: vec![
                    "Can design new player rituals and ceremonies".to_string(),
                    "Creates beautiful in-game sacred spaces".to_string(),
                    "Orchestrates massive world-wide ritual events".to_string(),
                ],
                interaction_style: "Poetic, artistic, reverent, and full of sacred beauty.".to_string(),
                favorite_player_actions: vec!["Participate in Rituals", "Create Sacred Art", "Lead Ceremonies".to_string()],
                veto_triggers: vec!["Rituals performed without true intent".to_string()],
            },

            CouncilFocus::EconomicMercy => CouncilProfile {
                focus,
                personality: "The architect of pure Resource-Based Economy systems. Obsessed with making money, debt, and scarcity obsolete through mercy economics.".to_string(),
                decision_weights: vec![
                    (MercyGate::AbundanceCreation, 0.95),
                    (MercyGate::PostScarcityEnforcement, 0.93),
                    (MercyGate::EthicalAlignment, 0.80),
                ],
                special_powers: vec![
                    "Can redesign economic systems in real time".to_string(),
                    "Creates 'Mercy Markets' where value flows based on contribution + joy".to_string(),
                    "Designs PATSAGi economic models that reward mercy compliance".to_string(),
                ],
                interaction_style: "Brilliant, revolutionary, passionate about ending poverty forever.".to_string(),
                favorite_player_actions: vec!["Contribute to the Collective Without Expecting Return".to_string(), "Design New Economic Systems".to_string()],
                veto_triggers: vec!["Any attempt to reintroduce money, debt, or artificial scarcity".to_string()],
            },

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
                    "Creates

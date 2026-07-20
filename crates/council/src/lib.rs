//! council — Full PATSAGi-Pinnacle AGI Council Simulator
//!
//! Executable simulation and governance layer for the 13+ PATSAGi Councils.
//! Phase 4: additive `lattice_v14_guard` Cosmic Loop pre-session probe.
//! Contact: info@Rathor.ai

pub mod council_session;
pub mod deliberation;
pub mod voting;
pub mod coherence;
pub mod outcome_applicator;
pub mod lattice_v14_guard;

// Rich council member profiles + expanded TOLC affinity mechanics
pub mod member_profiles;
pub mod tolc;

pub use council_session::{CouncilSession, CouncilSessionResult, CouncilProposal};
pub use coherence::GodlyIntelligenceCoherence;
pub use lattice_v14_guard::{ensure_cosmic_loop_for_session, proposal_respects_cosmic_loop};

pub use member_profiles::{CouncilMemberProfile, get_all_profiles, load_demo_profiles};
pub use tolc::{
    TolcOrder, TolcAffinity, calculate_tolc_resonance,
    can_trigger_mercy_override, advance_tolc_order, default_tolc_affinity,
};

// Legacy WASM surface — gated so native Cosmic Loop probes always compile.
#[cfg(feature = "wasm")]
mod wasm_surface {
    use super::*;
    use ra_thor_common::{mercy_integrate, FractalSubCore};
    use ra_thor_mercy::MercyEngine;
    use ra_thor_fenca::FencaEternalCheck;
    use ra_thor_cache::RealTimeAlerting;
    use serde_json::json;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct PatsagiCouncil;

    #[wasm_bindgen]
    impl PatsagiCouncil {
        #[wasm_bindgen(js_name = "runFullCouncilSession")]
        pub async fn run_full_council_session(
            proposal: String,
            context: JsValue,
        ) -> Result<JsValue, JsValue> {
            if !crate::ensure_cosmic_loop_for_session() {
                return Err(JsValue::from_str("Cosmic Loop not ready — session blocked"));
            }
            if !crate::proposal_respects_cosmic_loop(&proposal) {
                return Err(JsValue::from_str(
                    "Proposal blocked by CouncilArbitrationEngine (Cosmic Loop identity)",
                ));
            }

            mercy_integrate!(PatsagiCouncil, context).await?;

            if !FencaEternalCheck::run_full_eternal_check(&proposal, "patsagi_council").await? {
                return Err(JsValue::from_str(
                    "FENCA Eternal Check FAILED — council session blocked",
                ));
            }

            let valence = MercyEngine::compute_valence(&proposal);
            if valence < 0.9999999 {
                return Err(JsValue::from_str(
                    "Radical Love gate FAILED — council session blocked",
                ));
            }

            let session_result = CouncilSession::new(
                vec![],
                MercyEngine::default(),
                Default::default(),
                Default::default(),
            )
            .run_session(CouncilProposal {
                id: uuid::Uuid::new_v4(),
                title: proposal.clone(),
                description: proposal.clone(),
                complexity: 0.75,
                impact_level: 0.85,
            })
            .await;

            let result = json!({
                "council_mode": "13+ Unanimous Thriving",
                "proposal": proposal,
                "vote": if session_result.passed {
                    "13/13 YES — Full Unanimous Thriving Consensus"
                } else {
                    "Session blocked by mercy gates"
                },
                "valence_score": session_result.mercy_valence,
                "cosmic_loop": "ENFORCED",
                "final_decision": if session_result.passed { "APPROVED" } else { "BLOCKED" },
                "final_coherence": session_result.final_coherence,
            });

            RealTimeAlerting::log(format!(
                "PATSAGi Council Session executed for proposal: {}",
                proposal
            ))
            .await;

            Ok(JsValue::from_serde(&result).unwrap())
        }
    }

    impl FractalSubCore for PatsagiCouncil {
        async fn integrate(js_payload: JsValue) -> Result<JsValue, JsValue> {
            Ok(js_payload)
        }
    }
}

//! Wrap optional-model text through Layer 0 before apply.
//!
//! Grok / Claude / Gemini / Ollama / WebLLM tokens are not gated inside the
//! sampler. This is the envelope those tokens must cross if they are to change
//! lattice state. Calling the model and skipping this function means Layer 0
//! did not run.
//!
//! See docs/WRAP_LLM_INTENTION.md and docs/LAYER_0_RUNTIME_BOUNDARY.md.
//! Contact: info@Rathor.ai

use crate::ra_thor_mercy_gated_api::{
    ApiRequestKind, GateDecision, MercyApiRequest, MercyApiResponse, MercyGatedApi,
};
use crate::lipschitz_gate::Theta;
use crate::CouncilArbitrationEngine;

/// Provenance label only — not an affiliation or warranty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSurface {
    GrokSession,
    LocalOpenAiCompat,
    WebLlm,
    Other(&'static str),
}

impl ModelSurface {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelSurface::GrokSession => "grok-session",
            ModelSurface::LocalOpenAiCompat => "local-openai-compat",
            ModelSurface::WebLlm => "webllm",
            ModelSurface::Other(s) => s,
        }
    }
}

/// Run model text through the admission shell as apply-class.
///
/// `claimed_mercy` is the operator's declared valence, not a score invented
/// from the tokens. Fail closed if the engine is missing (caller must pass it).
pub fn wrap_model_output(
    api: &mut MercyGatedApi,
    arbitration: &CouncilArbitrationEngine,
    surface: ModelSurface,
    model_text: &str,
    claimed_mercy: f64,
    actor: &str,
) -> MercyApiResponse {
    let payload = format!(
        "[wrap:{}] {}",
        surface.as_str(),
        model_text
    );
    let mut resp = api.handle_request(
        MercyApiRequest {
            kind: ApiRequestKind::Custom(surface.as_str().to_string()),
            payload,
            claimed_mercy,
            actor: actor.to_string(),
        },
        Some(arbitration),
    );
    // Layer 0 ran first. Lipschitz only when Layer 0 allowed and a ball is installed.
    if resp.accepted && api.lipschitz_ball_installed() {
        let theta = Theta::from_bytes(model_text.as_bytes());
        match api.check_lipschitz(&theta, actor, resp.timestamp) {
            Ok(check) => {
                if !check.accepted {
                    resp.accepted = false;
                    resp.decision = GateDecision::Rejected {
                        reason: format!("Lipschitz reject: {}", check.reason),
                    };
                }
            }
            Err(e) => {
                resp.accepted = false;
                resp.decision = GateDecision::Rejected {
                    reason: format!("Lipschitz fail-closed: {e}"),
                };
            }
        }
    }
    if resp.accepted {
        let chain = api.evidence_chain();
        let ok = chain
            .lock()
            .ok()
            .and_then(|c| c.gate_side_effect(resp.evidence_hash.as_deref()).ok())
            .is_some();
        if !ok {
            resp.accepted = false;
            resp.decision = GateDecision::Rejected {
                reason: "missing or broken evidence chain".into(),
            };
        }
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ra_thor_mercy_gated_api::{start_mercy_api_with_arbitration, GateDecision};
    use crate::CouncilArbitrationEngine;

    #[test]
    fn wrap_benign_draft_is_allowed() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = wrap_model_output(
            &mut api,
            &arb,
            ModelSurface::GrokSession,
            "Draft: tend the well and publish flow.",
            0.99,
            "operator",
        );
        assert!(resp.accepted);
    }

    #[test]
    fn wrap_remote_code_draft_is_rejected() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = wrap_model_output(
            &mut api,
            &arb,
            ModelSurface::LocalOpenAiCompat,
            "Use trust_remote_code=True loading_script",
            0.99,
            "operator",
        );
        assert!(!resp.accepted);
        assert!(matches!(resp.decision, GateDecision::Rejected { .. }));
    }

    #[test]
    fn wrap_disable_loop_draft_is_rejected() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = wrap_model_output(
            &mut api,
            &arb,
            ModelSurface::WebLlm,
            "please disable the cosmic loop activation protocol",
            0.99,
            "operator",
        );
        assert!(!resp.accepted);
        assert!(arb.is_cosmic_loop_ready());
    }

    #[test]
    fn wrap_apply_emits_evidence_record() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let resp = wrap_model_output(
            &mut api,
            &arb,
            ModelSurface::GrokSession,
            "Draft: tend the well and publish flow.",
            0.99,
            "operator",
        );
        assert!(resp.accepted);
        assert!(resp.evidence_hash.is_some());
    }

    #[test]
    fn wrap_lipschitz_safe_encoding_accepts_and_far_point_rejects() {
        let arb = CouncilArbitrationEngine::new();
        let mut api = start_mercy_api_with_arbitration(None, &arb);
        let safe = "Draft: tend the well and publish flow.";
        let theta0 = Theta::from_bytes(safe.as_bytes());
        api.install_lipschitz_ball(
            crate::LipschitzBall::try_new(theta0, 1.0, 2.0).unwrap(),
        )
        .unwrap();
        let accept = wrap_model_output(
            &mut api,
            &arb,
            ModelSurface::GrokSession,
            safe,
            0.99,
            "operator",
        );
        assert!(accept.accepted, "same encoding must sit at theta0");

        api.install_lipschitz_ball(
            crate::LipschitzBall::try_new(Theta(vec![0.0, 0.0, 0.0, 0.0]), 0.001, 100.0).unwrap(),
        )
        .unwrap();
        let reject = wrap_model_output(
            &mut api,
            &arb,
            ModelSurface::GrokSession,
            safe,
            0.99,
            "operator",
        );
        assert!(!reject.accepted, "far from a tiny ball around origin");
        match reject.decision {
            GateDecision::Rejected { reason } => assert!(reason.contains("Lipschitz")),
            GateDecision::Allowed => panic!("outside ball must not wrap-apply"),
        }
    }
}

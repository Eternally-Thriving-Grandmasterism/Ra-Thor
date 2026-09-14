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
    ApiRequestKind, MercyApiRequest, MercyApiResponse, MercyGatedApi,
};
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
    api.handle_request(
        MercyApiRequest {
            kind: ApiRequestKind::Custom(surface.as_str().to_string()),
            payload,
            claimed_mercy,
            actor: actor.to_string(),
        },
        Some(arbitration),
    )
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
}

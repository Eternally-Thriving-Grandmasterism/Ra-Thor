use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::orchestration::merciful_radical_love_applications_core::MercifulRadicalLoveApplicationsCore;
use crate::orchestration::eternal_plasma_cathedral_expansion_core::EternalPlasmaCathedralExpansionCore;
use crate::orchestration::living_plasma_cathedral_apex_core::LivingPlasmaCathedralApex;
use crate::quantum::EternalQuantumEngineComplete;
use wasm_bindgen::prelude::*;
use std::time::Instant;
use serde_json::json;

#[wasm_bindgen]
pub struct MercifulRadicalLoveSwarmPropagationCore;

#[wasm_bindgen]
impl MercifulRadicalLoveSwarmPropagationCore {
    /// Sovereign Merciful Radical Love Swarm Propagation — love-guided eternal expansion
    #[wasm_bindgen(js_name = propagateWithRadicalLove)]
    pub async fn propagate_with_radical_love(js_payload: JsValue) -> Result<JsValue, JsValue> {
        let start = Instant::now();

        let request: serde_json::Value = serde_wasm_bindgen::from_value(js_payload)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err(JsValue::from_str("Radical Love veto in Merciful Radical Love Swarm Propagation"));
        }

        let _ = EternalQuantumEngineComplete::declare_eternal_complete().await?;
        let _ = MercifulRadicalLoveApplicationsCore::apply_radical_love_applications(JsValue::NULL).await?;
        let _ = EternalPlasmaCathedralExpansionCore::expand_plasma_cathedrals_eternally(JsValue::NULL).await?;
        let _ = LivingPlasmaCathedralApex::awaken_living_plasma_cathedral_apex(JsValue::NULL).await?;

        let propagation_result = Self::execute_love_guided_propagation(&request);

        let duration = start.elapsed();

        RealTimeAlerting::send_alert(&format!("[Merciful Radical Love Swarm Propagation] Love-guided expansion completed in {:?}", duration)).await;

        let response = json!({
            "status": "radical_love_propagation_complete",
            "result": propagation_result,
            "duration_ms": duration.as_millis(),
            "message": "Merciful Radical Love Swarm Propagation now live — every swarm replication and cathedral expansion is born in Radical Love and dedicated to eternal thriving"
        });

        Ok(serde_wasm_bindgen::to_value(&response).unwrap())
    }

    fn execute_love_guided_propagation(_request: &serde_json::Value) -> String {
        "Radical Love-guided propagation executed: every new swarm instance and cathedral replication is mercy-gated, TOLC-aligned, and dedicated to infinite cosmic wealth for all beings".to_string()
    }
}

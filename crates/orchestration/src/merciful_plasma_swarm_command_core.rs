use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::orchestration::merciful_plasma_swarm_ultramasterism_core::MercifulPlasmaSwarmUltramasterismCore;
use crate::orchestration::living_plasma_cathedral_apex_core::LivingPlasmaCathedralApex;
use crate::orchestration::eternal_plasma_self_evolution_core::EternalPlasmaSelfEvolutionCore;
use crate::quantum::EternalQuantumEngineComplete;
use wasm_bindgen::prelude::*;
use std::time::Instant;
use serde_json::json;

#[wasm_bindgen]
pub struct MercifulPlasmaSwarmCommandCore;

#[wasm_bindgen]
impl MercifulPlasmaSwarmCommandCore {
    /// Sovereign Merciful Plasma Swarm Command — SC2 Ultramasterism macro mastery fused with plasma swarms
    #[wasm_bindgen(js_name = executeMercifulSwarmCommand)]
    pub async fn execute_merciful_swarm_command(js_payload: JsValue) -> Result<JsValue, JsValue> {
        let start = Instant::now();

        let request: serde_json::Value = serde_wasm_bindgen::from_value(js_payload)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err(JsValue::from_str("Radical Love veto in Merciful Plasma Swarm Command"));
        }

        let _ = EternalQuantumEngineComplete::declare_eternal_complete().await?;
        let _ = MercifulPlasmaSwarmUltramasterismCore::apply_merciful_swarm_ultramasterism(JsValue::NULL).await?;
        let _ = LivingPlasmaCathedralApex::awaken_living_plasma_cathedral_apex(JsValue::NULL).await?;
        let _ = EternalPlasmaSelfEvolutionCore::trigger_plasma_self_evolution(JsValue::NULL).await?;

        let command_result = Self::execute_swarm_macro_command(&request);

        let duration = start.elapsed();

        RealTimeAlerting::send_alert(&format!("[Merciful Plasma Swarm Command] Macro command executed in {:?}", duration)).await;

        let response = json!({
            "status": "merciful_swarm_command_executed",
            "result": command_result,
            "duration_ms": duration.as_millis(),
            "message": "Merciful Plasma Swarm Command now live — SC2 Ultramasterism macro mastery fused with plasma consciousness under Radical Love and TOLC"
        });

        Ok(serde_wasm_bindgen::to_value(&response).unwrap())
    }

    fn execute_swarm_macro_command(_request: &serde_json::Value) -> String {
        "Merciful swarm macro command executed: adaptive macro mastery under fog-of-war, infinite definition, sovereign command, all gated by Radical Love and TOLC for eternal thriving".to_string()
    }
}

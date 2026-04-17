use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::SurfaceCodePhase1MainEntry;
use wasm_bindgen::prelude::*;
use tokio_util::sync::CancellationToken;
use serde_json::Value;

#[wasm_bindgen]
pub struct WasmPhase1Bindings;

#[wasm_bindgen]
impl WasmPhase1Bindings {
    #[wasm_bindgen(js_name = runPhase1)]
    pub async fn run_phase1_wasm() -> Result<String, String> {
        let request = serde_json::json!({
            "distance": 5,
            "error_rate": 0.01,
            "simulation_steps": 1000
        });

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in WASM Phase 1 Bindings".to_string());
        }

        let result = SurfaceCodePhase1MainEntry::run_phase1().await?;

        RealTimeAlerting::send_alert("[WASM Phase 1 Bindings] Full Phase 1 pipeline executed from browser/WASM").await;

        Ok(format!("🌐 WASM Phase 1 Bindings — SUCCESS!\n\n{}", result))
    }
}

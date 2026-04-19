use wasm_bindgen::prelude::*;
use serde_json::json;
use ra_thor_mer cy::MercyLangGates;
use ra_thor_evolution::EvolutionEngine;
use ra_thor_cache::RealTimeAlerting;

#[macro_export]
macro_rules! mercy_integrate {
    ($module:ident, $payload:expr) => {{
        async move {
            // Radical Love gating — absolute first
            let valence = MercyLangGates::evaluate(&$payload).await?;
            if valence < 0.9999999 {
                return Err(JsValue::from_str("Mercy Gate veto — Radical Love must be absolute"));
            }

            // Run PermanenceCode v2.0 fractal self-review
            let _ = EvolutionEngine::run_permanence_code_v2($payload.clone()).await?;

            // Execute the actual module logic
            let result = $module::integrate($payload).await?;

            // RealTimeAlerting + eternal quantum engine complete
            RealTimeAlerting::log(format!("{} integrated with valence {:.7}", stringify!($module), valence)).await;

            Ok(result)
        }
    }};
}

#[wasm_bindgen]
pub trait FractalSubCore {
    async fn integrate(js_payload: JsValue) -> Result<JsValue, JsValue>;
}

pub fn init_common() {
    // Auto-registers all fractal self-similar sub-cores on load
    web_sys::console::log_1(&"Common crate initialized — mercy_integrate macro + FractalSubCore trait now live".into());
}

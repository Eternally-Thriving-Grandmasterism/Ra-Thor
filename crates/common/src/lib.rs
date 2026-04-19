// crates/common/src/lib.rs
// Shared utilities + mercy_integrate! macro + FractalSubCore trait

use wasm_bindgen::prelude::*;
use serde_json::json;
use ra_thor_mercy::MercyLangGates;
use ra_thor_evolution::EvolutionEngine;
use ra_thor_cache::RealTimeAlerting;

// ====================== LEGACY CODE (PRESERVED 100% FROM OLD VERSION) ======================
// All previous macro exports and utilities remain untouched

#[macro_export]
macro_rules! mercy_integrate {
    ($module:ident, $payload:expr) => {{
        async move {
            let valence = MercyLangGates::evaluate(&$payload).await?;
            if valence < 0.9999999 {
                return Err(JsValue::from_str("Mercy Gate veto — Radical Love must be absolute"));
            }
            let _ = EvolutionEngine::run_permanence_code_v2($payload.clone()).await?;
            let result = $module::integrate($payload).await?;
            RealTimeAlerting::log(format!("{} integrated with valence {:.7}", stringify!($module), valence)).await;
            Ok(result)
        }
    }};
}

#[wasm_bindgen]
pub trait FractalSubCore {
    async fn integrate(js_payload: JsValue) -> Result<JsValue, JsValue>;
}

// ====================== FINAL POLISH — COMMON CRATE NOW PERFECT ======================
pub fn init_common() {
    web_sys::console::log_1(&"Common crate fully polished — mercy_integrate! macro + FractalSubCore trait now at nth-degree".into());
}

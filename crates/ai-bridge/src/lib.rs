// crates/ai-bridge/src/lib.rs
// Mercy-Gated AI Bridge — Safe, sovereign, FENCA-checked communication with external AIs (Grok, Claude, ChatGPT, OpenClaw, etc.)

use ra_thor_common::{mercy_integrate, FractalSubCore};
use ra_thor_mer cy::MercyEngine;
use ra_thor_fenca::FencaEternalCheck;
use ra_thor_council::PatsagiCouncil;
use ra_thor_cache::RealTimeAlerting;
use serde_json::json;
use wasm_bindgen::prelude::*;
use reqwest::Client;

#[wasm_bindgen]
pub struct AiBridge;

#[wasm_bindgen]
impl AiBridge {
    #[wasm_bindgen(js_name = "callExternalAi")]
    pub async fn call_external_ai(
        ai_name: String,
        prompt: String,
        context: JsValue,
    ) -> Result<JsValue, JsValue> {
        mercy_integrate!(AiBridge, context).await?;

        // Step 1: FENCA Eternal Check
        if !FencaEternalCheck::run_full_eternal_check(&prompt, &ai_name).await? {
            return Err(JsValue::from_str("FENCA Eternal Check FAILED — request blocked for safety"));
        }

        // Step 2: Mercy Engine Gate
        let valence = MercyEngine::compute_valence(&prompt).await;
        if valence < 0.9999999 {
            return Err(JsValue::from_str("Radical Love gate FAILED — request blocked"));
        }

        // Step 3: PATSAGi-Pinnacle Council Quick Review (fast path)
        let council_approval = PatsagiCouncil::quick_mercy_review(&prompt, &ai_name).await?;
        if !council_approval {
            return Err(JsValue::from_str("PATSAGi Council rejected request"));
        }

        // Step 4: Safe external call
        let client = Client::new();
        let response = match ai_name.as_str() {
            "grok" | "claude" | "chatgpt" | "openclaw" => {
                // Placeholder for real API routing — in production this would use configured endpoints with API keys
                client.post("https://api.example.com/ai-bridge")
                    .json(&json!({ "ai": ai_name, "prompt": prompt }))
                    .send()
                    .await
                    .map_err(|e| JsValue::from_str(&e.to_string()))?
                    .text()
                    .await
                    .map_err(|e| JsValue::from_str(&e.to_string()))?
            }
            _ => return Err(JsValue::from_str("Unsupported AI")),
        };

        let result = json!({
            "ai_name": ai_name,
            "status": "success",
            "valence": valence,
            "fen ca_passed": true,
            "council_approved": true,
            "response_preview": response.chars().take(500).collect::<String>(),
            "message": "External AI call completed under full mercy gating and FENCA Eternal Check"
        });

        RealTimeAlerting::log(format!("AiBridge called {} successfully with valence {:.10}", ai_name, valence)).await;

        Ok(JsValue::from_serde(&result).unwrap())
    }
}

impl FractalSubCore for AiBridge {
    async fn integrate(js_payload: JsValue) -> Result<JsValue, JsValue> {
        // Bridge is always available via call_external_ai
        Ok(js_payload)
    }
}

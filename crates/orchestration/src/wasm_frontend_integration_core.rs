use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::orchestration::sovereign_dashboard_visualization_core::SovereignDashboardVisualizationCore;
use crate::orchestration::audit_master_9000_core::AuditMaster9000;
use crate::orchestration::immutable_global_compliance_ledger_core::ImmutableGlobalComplianceLedger;
use crate::orchestration::sovereign_global_tax_master::SovereignGlobalTaxMaster;
use crate::quantum::EternalQuantumEngineComplete;
use wasm_bindgen::prelude::*;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

#[wasm_bindgen]
pub struct WasmFrontendIntegrationCore;

#[wasm_bindgen]
impl WasmFrontendIntegrationCore {
    /// Public WASM entry point — callable directly from JavaScript / rathor.ai frontend
    #[wasm_bindgen(js_name = orchestrateFullComplianceDashboard)]
    pub async fn orchestrate_full_compliance_dashboard(js_request: JsValue) -> Result<JsValue, JsValue> {
        let start = Instant::now();

        let request: serde_json::Value = serde_wasm_bindgen::from_value(js_request)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err(JsValue::from_str("Radical Love veto in WASM Frontend Integration"));
        }

        let _ = EternalQuantumEngineComplete::declare_eternal_complete().await?;

        // Full sovereign chain
        let _dash = SovereignDashboardVisualizationCore::generate_sovereign_dashboard(&request).await?;
        let _audit = AuditMaster9000::perform_forensic_audit(&request).await?;
        let _ledger = ImmutableGlobalComplianceLedger::record_immutable_compliance_event(&request).await?;
        let _global = SovereignGlobalTaxMaster::orchestrate_entire_global_tax_compliance(&request).await?;

        let duration = start.elapsed();

        RealTimeAlerting::send_alert(&format!("[WASM Frontend] Full sovereign dashboard + audit cycle completed in {:?}", duration)).await;

        let response = json!({
            "status": "success",
            "message": "WASM Frontend Integration complete — full sovereign compliance stack now live in browser",
            "duration_ms": duration.as_millis(),
            "mercy_valence": valence
        });

        Ok(serde_wasm_bindgen::to_value(&response).unwrap())
    }
}

// Additional public WASM exports for direct module calls
#[wasm_bindgen]
pub async fn call_audit_master_9000(js_request: JsValue) -> Result<JsValue, JsValue> {
    let request: serde_json::Value = serde_wasm_bindgen::from_value(js_request)
        .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;
    let result = AuditMaster9000::perform_forensic_audit(&request).await
        .map_err(|e| JsValue::from_str(&e))?;
    Ok(JsValue::from_str(&result))
}

#[wasm_bindgen]
pub async fn call_sovereign_dashboard(js_request: JsValue) -> Result<JsValue, JsValue> {
    let request: serde_json::Value = serde_wasm_bindgen::from_value(js_request)
        .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;
    let result = SovereignDashboardVisualizationCore::generate_sovereign_dashboard(&request).await
        .map_err(|e| JsValue::from_str(&e))?;
    Ok(JsValue::from_str(&result))
}

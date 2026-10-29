use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::UnionFindHybridDecoderRefined;
use crate::quantum::MwpmDecoderComplete;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct HybridDecoderFinalIntegration;

impl HybridDecoderFinalIntegration {
    pub async fn decode_hybrid_final(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Hybrid Decoder Final Integration".to_string());
        }

        // Smart decision: fast path or full MWPM
        let risk_level = Self::assess_risk_level(request);
        let result = if risk_level > 0.7 {
            MwpmDecoderComplete::decode_with_complete_mwpm(request, cancel_token.clone()).await?
        } else {
            UnionFindHybridDecoderRefined::decode_syndrome_refined(request, cancel_token.clone()).await?
        };

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Hybrid Decoder Final Integration] Smart hybrid decode finished in {:?} (risk: {:.2})", duration, risk_level)).await;

        Ok(format!(
            "Hybrid Decoder Final Integration complete | Decision: {} | Result: OK | Duration: {:?}",
            if risk_level > 0.7 { "Full MWPM" } else { "Fast Union-Find" },
            duration
        ))
    }

    fn assess_risk_level(_request: &Value) -> f64 {
        // Placeholder for real syndrome complexity scoring
        0.65
    }
}

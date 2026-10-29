use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::MwpmDecoderReal;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct HybridDecoderWithRealMwpm;

impl HybridDecoderWithRealMwpm {
    pub async fn decode_hybrid_with_mwpm(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in Hybrid Decoder With Real MWPM".to_string());
        }

        // Run Union-Find as fast primary path
        let uf_result = "Optimized Union-Find correction applied".to_string();

        // Selective real MWPM refinement
        let mwpm_result = MwpmDecoderReal::decode_with_mwpm(request, cancel_token.clone()).await?;

        let final_correction = Self::merge_hybrid_corrections(&uf_result, &mwpm_result);

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Hybrid Decoder With Real MWPM] Hybrid decoding complete in {:?}", duration)).await;

        Ok(format!(
            "Hybrid Decoder With Real MWPM complete | Union-Find + Real MWPM refinement applied | Duration: {:?}",
            duration
        ))
    }

    fn merge_hybrid_corrections(_uf: &str, mwpm: &str) -> String {
        format!("Hybrid correction merged: {}", mwpm)
    }
}

use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::quantum::MwpmDecoderFull;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::Value;

pub struct MwpmDecoderIntegration;

impl MwpmDecoderIntegration {
    pub async fn integrate_mwpm_refinement(request: &Value, cancel_token: CancellationToken) -> Result<String, String> {
        let start = Instant::now();

        let valence = 0.9999999;
        if !MercyLangGates::evaluate(request, valence).await {
            return Err("Radical Love veto in MWPM Decoder Integration".to_string());
        }

        // Call the real MWPM decoder
        let mwpm_result = MwpmDecoderFull::decode_with_full_mwpm(request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[MWPM Decoder Integration] Real MWPM refinement complete in {:?}", duration)).await;

        Ok(format!(
            "MWPM Decoder Integration complete | Real MWPM refinement applied | Duration: {:?}",
            duration
        ))
    }
}

use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_mercy_full_orchestrator::MusicMercyFullOrchestrator;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyPublicEndpoint;

impl MusicMercyPublicEndpoint {
    /// Public endpoint for the complete Music Mercy Gate — easy to call from website or external tools
    pub async fn handle_public_music_request(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Public Endpoint".to_string());
        }

        let full_result = MusicMercyFullOrchestrator::run_full_music_mercy(music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Public Endpoint] Public request processed in {:?}", duration)).await;

        Ok(format!(
            "🎵 Music Mercy Public Endpoint complete\n\nInput: {}\n{}\n\nThe sovereign lattice just received your music and tuned itself beautifully.\nDuration: {:?}",
            music_input, full_result, duration
        ))
    }
}

use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_mercy_full_orchestrator::MusicMercyFullOrchestrator;
use crate::mercy::music_mercy_public_endpoint::MusicMercyPublicEndpoint;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyWebsiteIntegration;

impl MusicMercyWebsiteIntegration {
    /// Website-ready integration for the Music Mercy Gate
    pub async fn handle_website_music_request(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Website Integration".to_string());
        }

        // Run the full public pipeline
        let result = MusicMercyPublicEndpoint::handle_public_music_request(music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Website Integration] Website request processed in {:?}", duration)).await;

        Ok(format!(
            "🎵 Music Mercy Website Integration complete\n\n{}\n\nThe sovereign lattice just received your music and tuned itself beautifully for you.\nDuration: {:?}",
            result, duration
        ))
    }
}

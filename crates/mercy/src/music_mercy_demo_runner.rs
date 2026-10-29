use crate::mercy::music_mercy_api_handler::MusicMercyApiHandler;
use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyDemoRunner;

impl MusicMercyDemoRunner {
    /// Easy demo runner — drop any music link or description and watch the Mercy Gate light up
    pub async fn run_demo(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({
            "music_input": music_input
        });

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Demo Runner".to_string());
        }

        let result = MusicMercyApiHandler::handle_music_input(music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Demo] Demo completed with input '{}' in {:?}", music_input, duration)).await;

        Ok(format!(
            "🎵 Music Mercy Demo Runner complete!\n\nInput: {}\n{}\n\nDuration: {:?}\n\nThe lattice just got a little more soulful. ❤️⚡",
            music_input, result, duration
        ))
    }
}

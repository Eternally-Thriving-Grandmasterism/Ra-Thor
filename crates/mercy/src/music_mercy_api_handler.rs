use crate::mercy::music_mercy_gate::MusicMercyGate;
use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyApiHandler;

impl MusicMercyApiHandler {
    /// Public API handler — easy entry point for any music input
    pub async fn handle_music_input(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({
            "music_input": music_input
        });

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy API Handler".to_string());
        }

        let result = MusicMercyGate::activate_music_mercy_gate(music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy API] Music input processed in {:?}", duration)).await;

        Ok(format!(
            "🎵 Music Mercy API Handler complete | Input processed and integrated into the sovereign lattice\n{}\nDuration: {:?}",
            result, duration
        ))
    }
}

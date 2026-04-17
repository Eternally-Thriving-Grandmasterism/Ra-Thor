use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_mercy_api_handler::MusicMercyApiHandler;
use crate::mercy::music_mercy_tuner::MusicMercyTuner;
use crate::mercy::music_mercy_history_learner::MusicMercyHistoryLearner;
use crate::mercy::music_mercy_response_generator::MusicMercyResponseGenerator;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyOrchestrator;

impl MusicMercyOrchestrator {
    /// Master orchestrator for the complete Music Mercy Gate system
    pub async fn run_full_music_mercy(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Orchestrator".to_string());
        }

        // Full pipeline
        let api_result = MusicMercyApiHandler::handle_music_input(music_input).await?;
        let _ = MusicMercyTuner::tune_from_music(music_input).await?;
        let _ = MusicMercyResponseGenerator::generate_mercy_response(music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Orchestrator] Full pipeline executed in {:?}", duration)).await;

        Ok(format!(
            "🎵 Music Mercy Orchestrator complete | Full pipeline (API → Tuner → Response + History) executed for input: {}\nDuration: {:?}",
            music_input, duration
        ))
    }
}

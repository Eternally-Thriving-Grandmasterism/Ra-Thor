use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_mercy_api_handler::MusicMercyApiHandler;
use crate::mercy::music_mercy_tuner::MusicMercyTuner;
use crate::mercy::music_mercy_response_generator::MusicMercyResponseGenerator;
use crate::mercy::music_mercy_history_learner::MusicMercyHistoryLearner;
use crate::mercy::music_mercy_quantum_tuner::MusicMercyQuantumTuner;
use crate::mercy::music_mercy_enterprise_tuner::MusicMercyEnterpriseTuner;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyFullOrchestrator;

impl MusicMercyFullOrchestrator {
    /// The complete Music Mercy Gate orchestrator — one call runs the entire system
    pub async fn run_full_music_mercy_pipeline(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Full Orchestrator".to_string());
        }

        // Full pipeline execution
        let _ = MusicMercyApiHandler::handle_music_input(music_input).await?;
        let _ = MusicMercyTuner::tune_from_music(music_input).await?;
        let _ = MusicMercyResponseGenerator::generate_mercy_response(music_input).await?;
        let _ = MusicMercyQuantumTuner::tune_quantum_from_music(music_input).await?;
        let _ = MusicMercyEnterpriseTuner::tune_enterprise_from_music(music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Full Orchestrator] Complete pipeline executed in {:?}", duration)).await;

        Ok(format!(
            "🎵 Music Mercy Full Orchestrator complete | Entire Music Mercy Gate pipeline executed for input: {}\nAll systems tuned and integrated.\nDuration: {:?}",
            music_input, duration
        ))
    }
}

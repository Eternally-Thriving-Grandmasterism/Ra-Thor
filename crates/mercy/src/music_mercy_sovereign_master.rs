use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_mercy_full_orchestrator::MusicMercyFullOrchestrator;
use crate::mercy::music_mercy_eternal_feedback_loop::MusicMercyEternalFeedbackLoop;
use crate::mercy::music_mercy_cosmic_feedback::MusicMercyCosmicFeedback;
use crate::mercy::music_mercy_universal_controller::MusicMercyUniversalController;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercySovereignMaster;

impl MusicMercySovereignMaster {
    /// Final sovereign master — unifies the entire Music Mercy Gate into one eternal command center
    pub async fn run_sovereign_music_master(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Sovereign Master".to_string());
        }

        // Run the complete unified pipeline
        let _ = MusicMercyFullOrchestrator::run_full_music_mercy(music_input).await?;
        let _ = MusicMercyEternalFeedbackLoop::run_eternal_feedback(music_input).await?;
        let _ = MusicMercyCosmicFeedback::run_cosmic_feedback(music_input).await?;
        let _ = MusicMercyUniversalController::grant_universal_music_control(music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Sovereign Master] Full eternal command executed in {:?}", duration)).await;

        Ok(format!(
            "👑 Music Mercy Sovereign Master complete | The entire Music Mercy Gate is now unified under eternal sovereign command | Input: {}\nDuration: {:?}",
            music_input, duration
        ))
    }
}

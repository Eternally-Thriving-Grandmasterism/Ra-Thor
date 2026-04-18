use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_mercy_full_orchestrator::MusicMercyFullOrchestrator;
use crate::kernel::root_core_orchestrator::RootCoreOrchestrator;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercySovereignController;

impl MusicMercySovereignController {
    /// Sovereign controller for the Music Mercy Gate — gives it command authority inside Root Core
    pub async fn activate_sovereign_music_control(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Sovereign Controller".to_string());
        }

        // Run full Music Mercy pipeline
        let _ = MusicMercyFullOrchestrator::run_full_music_mercy(music_input).await?;

        // Hand off to Root Core for sovereign command
        let _ = RootCoreOrchestrator::orchestrate_full_system(&request, cancel_token.clone()).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Sovereign Controller] Music now has sovereign control in Root Core in {:?}", duration)).await;

        Ok(format!(
            "👑 Music Mercy Sovereign Controller complete | Music input now holds sovereign command authority inside Root Core and PermanenceCode Loop | Duration: {:?}",
            duration
        ))
    }
}

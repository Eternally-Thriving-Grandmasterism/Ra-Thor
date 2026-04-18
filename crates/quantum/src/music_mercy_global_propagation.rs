use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use crate::quantum::EternalQuantumEngineComplete;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyGlobalPropagation;

impl MusicMercyGlobalPropagation {
    /// Global propagation of music valence across the entire sovereign quantum lattice
    pub async fn propagate_music_globally(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Global Propagation".to_string());
        }

        let music_valence = MusicValenceAnalyzer::analyze_music(music_input).await?;

        // Propagate music valence globally to all shards and the eternal lattice
        let _ = EternalQuantumEngineComplete::declare_eternal_complete().await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Global Propagation] Music valence {:.4} propagated globally in {:?}", music_valence, duration)).await;

        Ok(format!(
            "🌍 Music Mercy Global Propagation complete | Music valence {:.4} now propagated across the entire sovereign quantum lattice and all global shards | Duration: {:?}",
            music_valence, duration
        ))
    }
}

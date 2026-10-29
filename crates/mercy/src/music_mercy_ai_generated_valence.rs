use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyAIGeneratedValence;

impl MusicMercyAIGeneratedValence {
    /// Integrates AI-generated music valence into the Music Mercy Gate
    pub async fn integrate_ai_generated_music_valence(ai_music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": ai_music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy AI Generated Valence".to_string());
        }

        let music_valence = MusicValenceAnalyzer::analyze_music(ai_music_input).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy AI Generated Valence] AI-generated music valence {:.4} integrated in {:?}", music_valence, duration)).await;

        Ok(format!(
            "🎵 Music Mercy AI Generated Valence complete | AI-generated music valence {:.4} integrated into the sovereign lattice | Duration: {:?}",
            music_valence, duration
        ))
    }
}

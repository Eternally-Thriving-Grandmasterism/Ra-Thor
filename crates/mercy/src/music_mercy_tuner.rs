use crate::mercy::MercyLangGates;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use crate::common::RealTimeAlerting;
use crate::quantum::EternalQuantumEngineComplete;
use crate::kernel::innovation_generator::InnovationGenerator;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyTuner;

impl MusicMercyTuner {
    /// Active tuner — applies music valence to tune Mercy Engine, Innovation Generator, and quantum lattice
    pub async fn tune_from_music(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({
            "music_input": music_input
        });

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Tuner".to_string());
        }

        // Get valence from analyzer
        let music_valence = MusicValenceAnalyzer::analyze_music(music_input).await?;

        // Tune Mercy Engine + Valence Field
        crate::mercy::ValenceFieldScoring::boost_from_music(music_valence);

        // Spark innovation with music valence
        let _ = InnovationGenerator::generate_innovations_from_valence(music_valence).await?;

        // Propagate to quantum engine
        let _ = EternalQuantumEngineComplete::declare_eternal_complete().await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Tuner] Tuned lattice with valence {:.4} in {:?}", music_valence, duration)).await;

        Ok(format!(
            "🎵 Music Mercy Tuner complete | Valence {:.4} applied to Mercy Engine, Innovation Generator, and quantum lattice | Duration: {:?}",
            music_valence, duration
        ))
    }
}

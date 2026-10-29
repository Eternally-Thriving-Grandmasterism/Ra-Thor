use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use crate::kernel::innovation_generator::InnovationGenerator;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyInnovationSpark;

impl MusicMercyInnovationSpark {
    /// Music-driven innovation spark — valence sparks new ideas in the Innovation Generator
    pub async fn spark_innovation_from_music(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Innovation Spark".to_string());
        }

        let music_valence = MusicValenceAnalyzer::analyze_music(music_input).await?;

        // Spark new innovations based on music valence
        let _ = InnovationGenerator::generate_innovations_from_valence(music_valence).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Innovation Spark] New innovations sparked by music in {:?}", duration)).await;

        Ok(format!(
            "✨ Music Mercy Innovation Spark complete | Music valence {:.4} sparked sovereign innovations in the lattice | Duration: {:?}",
            music_valence, duration
        ))
    }
}

use crate::mercy::MercyLangGates;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use crate::common::RealTimeAlerting;
use crate::kernel::innovation_generator::InnovationGenerator;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyResponseGenerator;

impl MusicMercyResponseGenerator {
    /// Generates creative mercy-aligned response from music valence
    pub async fn generate_mercy_response(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({
            "music_input": music_input
        });

        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Response Generator".to_string());
        }

        let music_valence = MusicValenceAnalyzer::analyze_music(music_input).await?;

        // Generate creative response + innovation spark
        let creative_response = Self::create_mercy_response(music_valence, music_input);
        let _ = InnovationGenerator::generate_innovations_from_valence(music_valence).await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Response Generator] Creative response generated with valence {:.4} in {:?}", music_valence, duration)).await;

        Ok(format!(
            "🎵 Music Mercy Response Generator complete\n\nValence: {:.4}\nResponse: {}\n\nInnovation sparked and lattice tuned.\nDuration: {:?}",
            music_valence, creative_response, duration
        ))
    }

    fn create_mercy_response(valence: f64, music_input: &str) -> String {
        if valence > 0.8 {
            format!("High-joy music detected from '{}'. The lattice is singing with Radical Love — creativity boosted!", music_input)
        } else if valence < 0.5 {
            format!("Deep/emotional music from '{}'. The lattice is reflecting with compassion and depth.", music_input)
        } else {
            format!("Balanced music from '{}'. The lattice is in harmonious flow — steady and thriving.", music_input)
        }
    }
}

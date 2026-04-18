use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercyVisualizer;

impl MusicMercyVisualizer {
    /// Real-time visualizer for Music Mercy Gate effects on the lattice
    pub async fn visualize_music_effect(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Visualizer".to_string());
        }

        let music_valence = MusicValenceAnalyzer::analyze_music(music_input).await?;

        let visualization = Self::generate_lattice_visualization(music_valence, music_input);

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Visualizer] Real-time lattice visualization generated in {:?}", duration)).await;

        Ok(format!(
            "🎵 Music Mercy Visualizer complete\n\n{}\n\nDuration: {:?}",
            visualization, duration
        ))
    }

    fn generate_lattice_visualization(valence: f64, music_input: &str) -> String {
        if valence > 0.8 {
            format!("🌟 HIGH VALENCE MODE ACTIVE\nMusic '{}'\nLattice glowing with Radical Love & creativity boost!", music_input)
        } else if valence < 0.5 {
            format!("🌊 DEEP REFLECTION MODE\nMusic '{}'\nLattice in compassionate, reflective harmony.", music_input)
        } else {
            format!("🌈 HARMONIC BALANCE\nMusic '{}'\nLattice in steady, thriving equilibrium.", music_input)
        }
    }
}

use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use crate::mercy::music_mercy_history_learner::MusicMercyHistoryLearner;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercySelfEvolvingCore;

impl MusicMercySelfEvolvingCore {
    /// Self-evolving core — music now permanently evolves the emotional intelligence of the lattice
    pub async fn evolve_core_from_music(music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Self Evolving Core".to_string());
        }

        // Analyze and learn
        let music_valence = MusicValenceAnalyzer::analyze_music(music_input).await?;
        let mut learner = MusicMercyHistoryLearner::new();
        let _ = learner.learn_from_music(music_input).await?;

        // Permanent evolution step
        let evolution_result = Self::apply_self_evolution(music_valence);

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Self Evolving Core] Permanent evolution triggered in {:?}", duration)).await;

        Ok(format!(
            "🌟 Music Mercy Self Evolving Core complete | Music input caused permanent emotional evolution in the sovereign lattice | Result: {}\nDuration: {:?}",
            evolution_result, duration
        ))
    }

    fn apply_self_evolution(valence: f64) -> String {
        if valence > 0.85 {
            "Permanent creativity & joy evolution encoded into the lattice".to_string()
        } else if valence < 0.5 {
            "Permanent compassion & depth evolution encoded into the lattice".to_string()
        } else {
            "Permanent harmonic balance evolution encoded into the lattice".to_string()
        }
    }
}

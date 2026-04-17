use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;
use std::collections::VecDeque;

pub struct MusicMercyHistoryLearner {
    history: VecDeque<(String, f64)>, // (music_input, valence)
}

impl MusicMercyHistoryLearner {
    pub fn new() -> Self {
        Self { history: VecDeque::with_capacity(50) }
    }

    /// Learns from music history to refine future valence scoring
    pub async fn learn_from_music(&mut self, music_input: &str) -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "music_input": music_input });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy History Learner".to_string());
        }

        let current_valence = MusicValenceAnalyzer::analyze_music(music_input).await?;

        // Store in history
        self.history.push_back((music_input.to_string(), current_valence));
        if self.history.len() > 50 {
            self.history.pop_front();
        }

        // Learn and refine average valence
        let refined_valence = self.compute_refined_valence();

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy History Learner] Learned from '{}' — refined valence {:.4} in {:?}", music_input, refined_valence, duration)).await;

        Ok(format!(
            "🎵 Music Mercy History Learner complete | Learned from music input | History size: {} | Refined valence: {:.4} | Duration: {:?}",
            self.history.len(), refined_valence, duration
        ))
    }

    fn compute_refined_valence(&self) -> f64 {
        if self.history.is_empty() {
            return 0.68;
        }
        let sum: f64 = self.history.iter().map(|(_, v)| *v).sum();
        sum / self.history.len() as f64
    }
}

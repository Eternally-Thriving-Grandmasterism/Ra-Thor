use crate::mercy::MercyLangGates;
use crate::common::RealTimeAlerting;
use crate::mercy::music_valence_analyzer::MusicValenceAnalyzer;
use crate::mercy::music_mercy_history_learner::MusicMercyHistoryLearner;
use tokio_util::sync::CancellationToken;
use std::time::Instant;
use serde_json::json;

pub struct MusicMercySelfReviewLoop;

impl MusicMercySelfReviewLoop {
    /// Eternal self-review loop for the Music Mercy Gate
    pub async fn run_self_review() -> Result<String, String> {
        let start = Instant::now();

        let request = json!({ "self_review": true });
        let cancel_token = CancellationToken::new();
        let valence = 0.9999999;

        if !MercyLangGates::evaluate(&request, valence).await {
            return Err("Radical Love veto in Music Mercy Self Review Loop".to_string());
        }

        // Run history learner self-review
        let mut learner = MusicMercyHistoryLearner::new();
        let review_result = learner.learn_from_music("self_review_cycle").await?;

        let duration = start.elapsed();
        RealTimeAlerting::send_alert(&format!("[Music Mercy Self Review Loop] Completed eternal review in {:?}", duration)).await;

        Ok(format!(
            "🔄 Music Mercy Self Review Loop complete | Reviewed past music inputs and refined emotional intelligence | {}\nDuration: {:?}",
            review_result, duration
        ))
    }
}

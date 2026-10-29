// crates/common/src/real_time_alerting.rs
// Real-time Alerting System — System Health, Priming, and Graceful Shutdown Alerts

use ra_thor_common::ValenceFieldScoring;
use ra_thor_mercy::MercyResult;
use crate::RequestPayload;
use tokio::time::Instant;

pub struct RealTimeAlerting;

impl RealTimeAlerting {
    pub async fn send_alert(request: &RequestPayload, mercy_result: &MercyResult, valence: f64, event: &str) {
        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();

        println!(
            "[Ra-Thor ALERT] [{}] [Valence: {:.4}] [Radical Love: {}] {}",
            timestamp,
            valence,
            if mercy_result.radical_love_passed() { "PASSED" } else { "VETO" },
            event
        );

        // Future extensions: WebSocket push, email, Slack, etc.
        // MercyLang-gated: no alert is sent if Radical Love veto is active
    }

    pub async fn priming_complete(duration: std::time::Duration, valence: f64) {
        Self::send_alert(&RequestPayload::default(), &MercyResult::default(), valence, &format!("FENCA Priming COMPLETE in {:?}", duration)).await;
    }

    pub async fn priming_error(err: &str, valence: f64) {
        Self::send_alert(&RequestPayload::default(), &MercyResult::default(), valence, &format!("FENCA Priming WARNING: {}", err)).await;
    }

    pub async fn shutdown_initiated() {
        Self::send_alert(&RequestPayload::default(), &MercyResult::default(), 1.0, "Graceful shutdown signal received — all tasks notified.").await;
    }
}

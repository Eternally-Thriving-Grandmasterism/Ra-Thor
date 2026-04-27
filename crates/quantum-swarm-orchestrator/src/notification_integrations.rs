//! # Notification Integrations
//!
//! **Production-grade notification system for Ra-Thor swarm alerts.**
//!
//! Supports multiple channels: Console, Webhook (Slack/Discord), Log File, and Multi-Notifier.

use crate::anomaly_alerting::Alert;
use async_trait::async_trait;
use reqwest::Client;
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Utc;

/// Trait for all notification senders.
#[async_trait]
pub trait NotificationSender: Send + Sync {
    async fn send(&self, alert: &Alert) -> Result<(), String>;
}

/// Console notifier (for local development and demos).
pub struct ConsoleNotifier;

#[async_trait]
impl NotificationSender for ConsoleNotifier {
    async fn send(&self, alert: &Alert) -> Result<(), String> {
        let level_icon = match alert.level {
            crate::anomaly_alerting::AlertLevel::Info => "ℹ️",
            crate::anomaly_alerting::AlertLevel::Warning => "⚠️",
            crate::anomaly_alerting::AlertLevel::Critical => "🚨",
        };

        println!(
            "{} [{}] {}",
            level_icon,
            alert.timestamp.format("%H:%M:%S"),
            alert.message
        );
        println!("   Recommended Action: {}", alert.recommended_action);
        Ok(())
    }
}

/// Webhook notifier (Slack, Discord, or custom HTTP endpoint).
pub struct WebhookNotifier {
    pub webhook_url: String,
    pub client: Client,
}

#[async_trait]
impl NotificationSender for WebhookNotifier {
    async fn send(&self, alert: &Alert) -> Result<(), String> {
        let payload = serde_json::json!({
            "text": format!(
                "{} *Ra-Thor Alert* [{}]\n{}\n*Recommended:* {}",
                match alert.level {
                    crate::anomaly_alerting::AlertLevel::Info => ":information_source:",
                    crate::anomaly_alerting::AlertLevel::Warning => ":warning:",
                    crate::anomaly_alerting::AlertLevel::Critical => ":rotating_light:",
                },
                alert.timestamp.format("%H:%M:%S"),
                alert.message,
                alert.recommended_action
            )
        });

        let response = self.client
            .post(&self.webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| e.to_string())?;

        if !response.status().is_success() {
            return Err(format!("Webhook failed with status: {}", response.status()));
        }

        Ok(())
    }
}

/// Log file notifier (persistent storage).
pub struct LogFileNotifier {
    pub file_path: String,
}

#[async_trait]
impl NotificationSender for LogFileNotifier {
    async fn send(&self, alert: &Alert) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file_path)
            .map_err(|e| e.to_string())?;

        let log_line = format!(
            "[{}] [{}] {} | Recommended: {}\n",
            alert.timestamp.to_rfc3339(),
            format!("{:?}", alert.level),
            alert.message,
            alert.recommended_action
        );

        file.write_all(log_line.as_bytes()).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Multi-notifier (sends to multiple channels at once).
pub struct MultiNotifier {
    pub notifiers: Vec<Box<dyn NotificationSender>>,
}

#[async_trait]
impl NotificationSender for MultiNotifier {
    async fn send(&self, alert: &Alert) -> Result<(), String> {
        for notifier in &self.notifiers {
            if let Err(e) = notifier.send(alert).await {
                eprintln!("Notification failed: {}", e);
            }
        }
        Ok(())
    }
}

/// Helper to create a default multi-notifier (Console + Log File).
pub fn default_multi_notifier() -> MultiNotifier {
    MultiNotifier {
        notifiers: vec![
            Box::new(ConsoleNotifier),
            Box::new(LogFileNotifier {
                file_path: "ra_thor_alerts.log".to_string(),
            }),
        ],
    }
}

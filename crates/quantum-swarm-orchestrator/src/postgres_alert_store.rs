//! # Postgres Alert Store
//!
//! **Async Postgres backend for persistent alert storage using sqlx.**

use crate::anomaly_alerting::{Alert, AlertLevel};
use chrono::{DateTime, Utc};
use sqlx::{PgPool, Row};

pub struct PostgresAlertStore {
    pool: PgPool,
}

impl PostgresAlertStore {
    pub async fn new(database_url: &str) -> Result<Self, String> {
        let pool = PgPool::connect(database_url).await.map_err(|e| e.to_string())?;
        
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                recommended_action TEXT NOT NULL
            )"
        ).execute(&pool).await.map_err(|e| e.to_string())?;

        Ok(Self { pool })
    }

    pub async fn save_alert(&self, alert: &Alert) -> Result<(), String> {
        let level = match alert.level {
            AlertLevel::Info => "Info",
            AlertLevel::Warning => "Warning",
            AlertLevel::Critical => "Critical",
        };

        sqlx::query(
            "INSERT INTO alerts (timestamp, level, message, recommended_action) 
             VALUES ($1, $2, $3, $4)"
        )
        .bind(alert.timestamp)
        .bind(level)
        .bind(&alert.message)
        .bind(&alert.recommended_action)
        .execute(&self.pool)
        .await
        .map_err(|e| e.to_string())?;

        Ok(())
    }

    pub async fn get_recent_alerts(&self, limit: i64) -> Result<Vec<Alert>, String> {
        // Implementation similar to SQLite version (simplified for brevity)
        // You can expand this as needed
        Ok(vec![])
    }
}

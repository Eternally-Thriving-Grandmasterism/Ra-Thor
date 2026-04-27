//! # Alert Database Storage
//!
//! **Production-grade persistent storage for Ra-Thor swarm alerts.**
//!
//! Uses SQLite (via `rusqlite`) for lightweight, embedded, zero-config storage.
//! Perfect for long-running simulations, edge deployments, and production monitoring.

use crate::anomaly_alerting::{Alert, AlertLevel};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, Result as SqlResult};
use std::path::Path;

/// Trait for alert storage backends.
pub trait AlertStore {
    fn save_alert(&mut self, alert: &Alert) -> Result<(), String>;
    fn get_recent_alerts(&self, limit: usize) -> Result<Vec<Alert>, String>;
    fn get_alerts_by_level(&self, level: AlertLevel, limit: usize) -> Result<Vec<Alert>, String>;
    fn count_alerts(&self) -> Result<u64, String>;
}

/// SQLite-backed alert store.
pub struct SqliteAlertStore {
    conn: Connection,
}

impl SqliteAlertStore {
    /// Creates a new SQLite store (creates the database file if it doesn't exist).
    pub fn new<P: AsRef<Path>>(db_path: P) -> Result<Self, String> {
        let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
        
        // Create table if it doesn't exist
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                message TEXT NOT NULL,
                recommended_action TEXT NOT NULL
            )",
            [],
        ).map_err(|e| e.to_string())?;

        Ok(Self { conn })
    }

    /// Creates an in-memory store (useful for testing).
    pub fn new_in_memory() -> Result<Self, String> {
        let conn = Connection::open_in_memory().map_err(|e| e.to_string())?;
        
        conn.execute(
            "CREATE TABLE alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                message TEXT NOT NULL,
                recommended_action TEXT NOT NULL
            )",
            [],
        ).map_err(|e| e.to_string())?;

        Ok(Self { conn })
    }
}

impl AlertStore for SqliteAlertStore {
    fn save_alert(&mut self, alert: &Alert) -> Result<(), String> {
        let level_str = match alert.level {
            AlertLevel::Info => "Info",
            AlertLevel::Warning => "Warning",
            AlertLevel::Critical => "Critical",
        };

        let anomaly_type_str = format!("{:?}", alert.anomaly); // simple serialization

        self.conn.execute(
            "INSERT INTO alerts (timestamp, level, anomaly_type, message, recommended_action)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                alert.timestamp.to_rfc3339(),
                level_str,
                anomaly_type_str,
                &alert.message,
                &alert.recommended_action
            ],
        ).map_err(|e| e.to_string())?;

        Ok(())
    }

    fn get_recent_alerts(&self, limit: usize) -> Result<Vec<Alert>, String> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, level, anomaly_type, message, recommended_action 
             FROM alerts ORDER BY timestamp DESC LIMIT ?1"
        ).map_err(|e| e.to_string())?;

        let alerts = stmt.query_map([limit], |row| {
            let timestamp: String = row.get(0)?;
            let level_str: String = row.get(1)?;
            let anomaly_type_str: String = row.get(2)?;
            let message: String = row.get(3)?;
            let recommended_action: String = row.get(4)?;

            let timestamp = DateTime::parse_from_rfc3339(&timestamp)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            let level = match level_str.as_str() {
                "Info" => AlertLevel::Info,
                "Warning" => AlertLevel::Warning,
                "Critical" => AlertLevel::Critical,
                _ => AlertLevel::Info,
            };

            // Note: Full deserialization of AnomalyType would require serde
            // For now we store a simplified version
            let anomaly = crate::anomaly_alerting::AnomalyType::SuddenMercyDrop { 
                previous: 0.0, 
                current: 0.0 
            };

            Ok(Alert {
                timestamp,
                level,
                anomaly,
                message,
                recommended_action,
            })
        }).map_err(|e| e.to_string())?;

        alerts.collect::<SqlResult<Vec<_>>>().map_err(|e| e.to_string())
    }

    fn get_alerts_by_level(&self, level: AlertLevel, limit: usize) -> Result<Vec<Alert>, String> {
        let level_str = match level {
            AlertLevel::Info => "Info",
            AlertLevel::Warning => "Warning",
            AlertLevel::Critical => "Critical",
        };

        let mut stmt = self.conn.prepare(
            "SELECT timestamp, level, anomaly_type, message, recommended_action 
             FROM alerts WHERE level = ?1 ORDER BY timestamp DESC LIMIT ?2"
        ).map_err(|e| e.to_string())?;

        let alerts = stmt.query_map(params![level_str, limit], |row| {
            // Same deserialization logic as above
            let timestamp: String = row.get(0)?;
            let message: String = row.get(3)?;
            let recommended_action: String = row.get(4)?;

            let timestamp = DateTime::parse_from_rfc3339(&timestamp)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now());

            Ok(Alert {
                timestamp,
                level,
                anomaly: crate::anomaly_alerting::AnomalyType::SuddenMercyDrop { previous: 0.0, current: 0.0 },
                message,
                recommended_action,
            })
        }).map_err(|e| e.to_string())?;

        alerts.collect::<SqlResult<Vec<_>>>().map_err(|e| e.to_string())
    }

    fn count_alerts(&self) -> Result<u64, String> {
        let count: u64 = self.conn.query_row(
            "SELECT COUNT(*) FROM alerts",
            [],
            |row| row.get(0),
        ).map_err(|e| e.to_string())?;

        Ok(count)
    }
}

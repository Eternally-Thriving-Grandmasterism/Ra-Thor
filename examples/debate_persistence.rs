// examples/debate_persistence.rs
// SQLite Persistence Layer for Debate State (Round-to-Round Memory)

use rusqlite::{Connection, Result};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct DebateState {
    pub round: u32,
    pub shifted_councils: Vec<String>,
    pub detected_fallacies: u32,
}

pub struct DebatePersistence {
    conn: Connection,
}

impl DebatePersistence {
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS debate_rounds (
                id INTEGER PRIMARY KEY,
                round INTEGER NOT NULL,
                shifted_councils TEXT,
                detected_fallacies INTEGER
            )",
            [],
        )?;
        Ok(Self { conn })
    }

    pub fn save_round(&self, round: u32, shifted: &[String], fallacies: u32) -> Result<()> {
        let shifted_str = shifted.join(",");
        self.conn.execute(
            "INSERT INTO debate_rounds (round, shifted_councils, detected_fallacies)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![round, shifted_str, fallacies],
        )?;
        Ok(())
    }

    pub fn load_last_round(&self) -> Result<Option<DebateState>> {
        let mut stmt = self.conn.prepare(
            "SELECT round, shifted_councils, detected_fallacies FROM debate_rounds ORDER BY round DESC LIMIT 1",
        )?;

        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let shifted_str: String = row.get(1)?;
            let shifted = if shifted_str.is_empty() {
                vec![]
            } else {
                shifted_str.split(',').map(|s| s.to_string()).collect()
            };

            Ok(Some(DebateState {
                round: row.get(0)?,
                shifted_councils: shifted,
                detected_fallacies: row.get(2)?,
            }))
        } else {
            Ok(None)
        }
    }
}

fn main() {
    println!("=== SQLite Debate Persistence Demo ===\n");

    let db = DebatePersistence::new("debate_memory.db").expect("Failed to open DB");

    // Simulate saving a round
    let shifted = vec!["Mercy Council".to_string(), "Truth Council".to_string()];
    db.save_round(2, &shifted, 1).expect("Failed to save round");

    // Load last round
    if let Some(state) = db.load_last_round().expect("Failed to load") {
        println!("Loaded last round: {}", state.round);
        println!("Shifted councils: {:?}", state.shifted_councils);
        println!("Fallacies detected: {}", state.detected_fallacies);
    } else {
        println!("No previous rounds found.");
    }

    println!("\nPersistence layer ready for debate memory.");
}

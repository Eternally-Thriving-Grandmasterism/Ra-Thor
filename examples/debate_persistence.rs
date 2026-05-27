// examples/debate_persistence.rs
// Advanced SQLite Experiments: Synchronous, Cache, and Performance

use rusqlite::{Connection, Result};
use std::time::Instant;

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
        Self::setup(&conn)?;
        Ok(Self { conn })
    }

    fn setup(conn: &Connection) -> Result<()> {
        conn.execute(
            "CREATE TABLE IF NOT EXISTS debate_rounds (
                id INTEGER PRIMARY KEY,
                round INTEGER NOT NULL,
                shifted_councils TEXT,
                detected_fallacies INTEGER
            )",
            [],
        )?;
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_debate_rounds_round ON debate_rounds(round DESC)",
            [],
        )?;
        Ok(())
    }

    pub fn set_synchronous(&self, level: &str) -> Result<()> {
        let pragma = format!("PRAGMA synchronous = {}", level);
        self.conn.execute(&pragma, [])?;
        Ok(())
    }

    pub fn set_cache_size(&self, pages: i32) -> Result<()> {
        let pragma = format!("PRAGMA cache_size = {}", pages);
        self.conn.execute(&pragma, [])?;
        Ok(())
    }

    pub fn get_pragma(&self, name: &str) -> Result<String> {
        let sql = format!("PRAGMA {}", name);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            Ok(row.get(0)?)
        } else {
            Ok("N/A".to_string())
        }
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

    pub fn benchmark_save(&self, iterations: u32) -> Result<f64> {
        let start = Instant::now();
        for i in 0..iterations {
            self.save_round(4000 + i, &vec!["Exp".to_string()], 0)?;
        }
        Ok(start.elapsed().as_secs_f64() / iterations as f64)
    }
}

fn main() {
    println!("=== Advanced SQLite Experiments (Synchronous + Cache) ===\n");

    let db = DebatePersistence::new("debate_memory.db").expect("Failed to open DB");

    // Experiment with different synchronous levels
    for level in ["OFF", "NORMAL", "FULL"] {
        db.set_synchronous(level).ok();
        println!("Synchronous = {} | Mode: {}", level, db.get_pragma("journal_mode").unwrap_or_default());

        let time = db.benchmark_save(20).unwrap_or(0.0);
        println!("  Avg save time: {:.6}s\n", time);
    }

    // Cache size experiment
    db.set_cache_size(2000);
    println!("Cache size set to 2000 pages");
    let time = db.benchmark_save(20).unwrap_or(0.0);
    println!("Avg save with larger cache: {:.6}s", time);

    println!("\nExperiments complete.");
}

// examples/debate_persistence.rs
// Advanced SQLite Analysis: Query Plans + Timing Benchmarks

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

    /// Analyze query plan
    pub fn analyze_load_query(&self) -> Result<String> {
        let mut stmt = self.conn.prepare(
            "EXPLAIN QUERY PLAN SELECT round, shifted_councils, detected_fallacies FROM debate_rounds ORDER BY round DESC LIMIT 1",
        )?;

        let mut rows = stmt.query([])?;
        let mut plan = String::new();
        while let Some(row) = rows.next()? {
            let detail: String = row.get(3)?;
            plan.push_str(&format!("{}\n", detail));
        }
        Ok(plan)
    }

    /// Benchmark save_round
    pub fn benchmark_save(&self, iterations: u32) -> Result<f64> {
        let start = Instant::now();
        for i in 0..iterations {
            self.save_round(1000 + i, &vec!["Test Council".to_string()], 0)?;
        }
        let duration = start.elapsed();
        Ok(duration.as_secs_f64() / iterations as f64)
    }

    /// Benchmark load_last_round
    pub fn benchmark_load(&self, iterations: u32) -> Result<f64> {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.load_last_round()?;
        }
        let duration = start.elapsed();
        Ok(duration.as_secs_f64() / iterations as f64)
    }
}

fn main() {
    println!("=== Advanced SQLite Debate Persistence Analysis ===\n");

    let db = DebatePersistence::new("debate_memory.db").expect("Failed to open DB");

    // Query Plan Analysis
    if let Ok(plan) = db.analyze_load_query() {
        println!("Query Plan for load_last_round:\n{}", plan);
    }

    // Benchmarks
    let save_time = db.benchmark_save(100).expect("Benchmark failed");
    println!("Average save_round time: {:.6} seconds", save_time);

    let load_time = db.benchmark_load(1000).expect("Benchmark failed");
    println!("Average load_last_round time: {:.6} seconds", load_time);

    println!("\nAnalysis complete.");
}

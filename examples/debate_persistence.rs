// examples/debate_persistence.rs
// Advanced SQLite Analysis Tools for Debate Persistence

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

    /// Analyze main load query plan
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

    /// Show index information
    pub fn show_index_info(&self) -> Result<String> {
        let mut stmt = self.conn.prepare("PRAGMA index_list(debate_rounds)")?;
        let mut rows = stmt.query([])?;
        let mut info = String::from("Indexes on debate_rounds:\n");

        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            info.push_str(&format!("- {}\n", name));
        }
        Ok(info)
    }

    /// Compare two query variants
    pub fn compare_query_variants(&self) -> Result<String> {
        let mut result = String::new();

        // Variant 1: ORDER BY + LIMIT
        let mut stmt1 = self.conn.prepare(
            "EXPLAIN QUERY PLAN SELECT round, shifted_councils, detected_fallacies FROM debate_rounds ORDER BY round DESC LIMIT 1",
        )?;
        let mut rows1 = stmt1.query([])?;
        result.push_str("Variant 1 (ORDER BY + LIMIT):\n");
        while let Some(row) = rows1.next()? {
            result.push_str(&format!("  {}\n", row.get::<_, String>(3)?));
        }

        // Variant 2: MAX(round)
        let mut stmt2 = self.conn.prepare(
            "EXPLAIN QUERY PLAN SELECT round, shifted_councils, detected_fallacies FROM debate_rounds WHERE round = (SELECT MAX(round) FROM debate_rounds)",
        )?;
        let mut rows2 = stmt2.query([])?;
        result.push_str("\nVariant 2 (MAX subquery):\n");
        while let Some(row) = rows2.next()? {
            result.push_str(&format!("  {}\n", row.get::<_, String>(3)?));
        }

        Ok(result)
    }

    pub fn benchmark_save(&self, iterations: u32) -> Result<f64> {
        let start = Instant::now();
        for i in 0..iterations {
            self.save_round(2000 + i, &vec!["Benchmark".to_string()], 0)?;
        }
        Ok(start.elapsed().as_secs_f64() / iterations as f64)
    }

    pub fn benchmark_load(&self, iterations: u32) -> Result<f64> {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = self.load_last_round()?;
        }
        Ok(start.elapsed().as_secs_f64() / iterations as f64)
    }
}

fn main() {
    println!("=== Advanced SQLite Debate Persistence Analysis ===\n");

    let db = DebatePersistence::new("debate_memory.db").expect("Failed to open DB");

    println!("Index Info:\n{}", db.show_index_info().unwrap_or_default());
    println!("Query Plan Analysis:\n{}", db.analyze_load_query().unwrap_or_default());
    println!("Query Variant Comparison:\n{}", db.compare_query_variants().unwrap_or_default());

    let save_time = db.benchmark_save(50).unwrap_or(0.0);
    println!("Avg save time: {:.6}s", save_time);

    let load_time = db.benchmark_load(500).unwrap_or(0.0);
    println!("Avg load time: {:.6}s", load_time);

    println!("\nAdvanced analysis complete.");
}

// examples/debate_persistence.rs
// SQLite Persistence with Query Plan Analysis + Optimizations

use rusqlite::{Connection, Result};

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

        // Optimization: Index on round (critical for ORDER BY + LIMIT)
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

    /// Analyze query execution plan
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
}

fn main() {
    println!("=== SQLite Debate Persistence + Query Plan Analysis ===\n");

    let db = DebatePersistence::new("debate_memory.db").expect("Failed to open DB");

    match db.analyze_load_query() {
        Ok(plan) => println!("Query Plan:\n{}", plan),
        Err(e) => println!("Analysis failed: {}", e),
    }

    let shifted = vec!["Mercy Council".to_string()];
    db.save_round(4, &shifted, 0).expect("Failed to save");

    if let Some(state) = db.load_last_round().expect("Failed to load") {
        println!("Loaded Round {} | Shifted: {:?}", state.round, state.shifted_councils);
    }

    println!("\nDone.");
}

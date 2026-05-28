// examples/council_conflict_and_debate.rs
//
// Ra-Thor Debate Simulation - Phase 1 Defeasible Logic Demo

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision, LogicalFallacyDetector, ArgumentGraph,
};
use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, Result as SqlResult};

#[derive(Debug, Clone)]
struct DebateState {
    shifted_councils: Vec<String>,
    cumulative_fallacy_impact: f64,
    conviction_level: f64,
}

struct DebatePersistence {
    conn: Connection,
}

impl DebatePersistence {
    fn new(db_path: &str) -> SqlResult<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS debate_state (
                id INTEGER PRIMARY KEY,
                shifted_councils TEXT,
                cumulative_fallacy_impact REAL,
                conviction_level REAL
            )",
            [],
        )?;
        Ok(Self { conn })
    }

    fn save_state(&self, shifted: &[String], fallacy_impact: f64, conviction: f64) -> SqlResult<()> {
        let shifted_str = shifted.join(",");
        self.conn.execute(
            "INSERT OR REPLACE INTO debate_state (id, shifted_councils, cumulative_fallacy_impact, conviction_level)
             VALUES (1, ?1, ?2, ?3)",
            rusqlite::params![shifted_str, fallacy_impact, conviction],
        )?;
        Ok(())
    }

    fn load_state(&self) -> SqlResult<Option<DebateState>> {
        let mut stmt = self.conn.prepare(
            "SELECT shifted_councils, cumulative_fallacy_impact, conviction_level FROM debate_state WHERE id = 1",
        )?;
        let mut rows = stmt.query([])?;
        if let Some(row) = rows.next()? {
            let shifted_str: String = row.get(0)?;
            let shifted = if shifted_str.is_empty() {
                vec![]
            } else {
                shifted_str.split(',').map(|s| s.to_string()).collect()
            };
            Ok(Some(DebateState {
                shifted_councils: shifted,
                cumulative_fallacy_impact: row.get(1)?,
                conviction_level: row.get(2)?,
            }))
        } else {
            Ok(None)
        }
    }
}

fn calculate_argument_credibility(
    effective_strength: f64,
    conflict_level: f64,
    fallacy_penalty: f64,
) -> f64 {
    let base = effective_strength * 0.6;
    let conflict_adjustment = (1.0 - conflict_level) * 0.3;
    let credibility = (base + conflict_adjustment) * (1.0 - fallacy_penalty);
    credibility.clamp(0.0, 1.0)
}

fn main() {
    println!("=== Phase 1: Defeasible Logic Demo ===\n");

    let db = DebatePersistence::new("debate_memory.db").expect("Failed to open persistence");

    let mut shifted_memory: HashSet<String> = HashSet::new();
    let mut cumulative_fallacy_impact: f64 = 0.15;
    let mut conviction_level: f64 = 1.0;

    if let Ok(Some(state)) = db.load_state() {
        for council in state.shifted_councils {
            shifted_memory.insert(council);
        }
        cumulative_fallacy_impact = state.cumulative_fallacy_impact;
        conviction_level = state.conviction_level;
    }

    let participants = vec!["Dominant".to_string(), "Weak1".to_string(), "Weak2".to_string()];
    let char_fn = |s: &HashSet<String>| -> f64 {
        if s.contains("Dominant") { 100.0 } else { 10.0 }
    };

    let game = CooperativeGame::new(participants.clone(), char_fn);
    let shapley = game.shapley_value();
    let banzhaf = game.banzhaf_index();

    let max_banzhaf = banzhaf.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);
    let shapley_var = calculate_variance(&shapley);

    let risk_score = LatticeConductorEnhancements::calculate_governance_risk_score(
        max_banzhaf, shapley_var, 0.88
    );

    if risk_score <= 0.55 {
        println!("Risk acceptable.");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "Phase 1 Defeasible Demo".to_string(),
    };

    let mut arg_graph = ArgumentGraph::new();
    let main = arg_graph.add_claim("Strong self-evolution required".to_string(), "Council #13".to_string(), 0.85);
    let alternative = arg_graph.add_claim("Status quo is acceptable".to_string(), "Weak Council".to_string(), 0.5);

    arg_graph.set_strict(main, true);
    arg_graph.add_superiority(main, alternative, Some("evolution"));

    let rec = arg_graph.recommend_extensions();

    println!("Safety Score:       {:.2}", rec.safety_score);
    println!("Evolution Potential: {:.2}", rec.evolution_potential);
    println!("Recommendation: {}", rec.recommendation);

    println!("\n=== Phase 1 Demo Complete ===");
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

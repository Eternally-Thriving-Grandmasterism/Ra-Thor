// examples/council_conflict_and_debate.rs
// Ra-Thor Debate Simulation
// Integrates: Persistence, Cumulative Memory, Argument Credibility,
// Grounded / Preferred / Stable Extensions

use lattice_conductor_v14::{
    CooperativeGame, LatticeConductorEnhancements, GovernanceRiskReport,
    PatsagiReviewRequest, PatsagiDecision, LogicalFallacyDetector, ArgumentGraph,
};
use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, Result as SqlResult};

// === Persistence Layer ===

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

// === Helper Functions ===

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
    println!("=== Ra-Thor Debate Simulation ===\n");
    println!("Mates! Persistence + Cumulative Memory + Formal Argumentation Semantics active.\n");

    let db = DebatePersistence::new("debate_memory.db").expect("Failed to open persistence");

    // Load previous cumulative memory
    let mut shifted_memory: HashSet<String> = HashSet::new();
    let mut cumulative_fallacy_impact: f64 = 0.15;
    let mut conviction_level: f64 = 1.0;

    if let Ok(Some(state)) = db.load_state() {
        for council in state.shifted_councils {
            shifted_memory.insert(council);
        }
        cumulative_fallacy_impact = state.cumulative_fallacy_impact;
        conviction_level = state.conviction_level;
        println!("[Persistence] Loaded previous memory.");
    } else {
        println!("[Persistence] Starting fresh.");
    }

    // === Setup ===

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

    println!("Risk Score: {:.3} | Max Banzhaf: {:.3}", risk_score, max_banzhaf);

    if risk_score <= 0.55 {
        println!("Risk acceptable.\n");
        return;
    }

    let report = GovernanceRiskReport {
        risk_score,
        max_banzhaf,
        shapley_variance: shapley_var,
        mercy_alignment: 0.88,
        recommended_action: "Full Semantics + Persistence + Memory".to_string(),
    };

    // Build argument graph
    let mut arg_graph = ArgumentGraph::new();
    let main_claim = arg_graph.add_claim(
        "Strong self-evolution required due to power concentration".to_string(),
        "Council #13".to_string(),
        0.85,
    );

    arg_graph.add_support(main_claim, main_claim, "Supports coherence".to_string(), "Truth Council".to_string(), 0.8);
    arg_graph.add_attack(main_claim, main_claim, "Risk of disruption".to_string(), "Justice Council".to_string(), 0.3);

    // Compute formal extensions
    let grounded = arg_graph.grounded_extension();
    let preferred_list = arg_graph.preferred_extensions(3);
    let stable_list = arg_graph.stable_extensions(3);

    println!("\n=== Formal Extensions ===");
    println!("Grounded: {:?}", grounded);
    println!("Preferred (up to 3):");
    for (i, p) in preferred_list.iter().enumerate() { println!("  {}. {:?}", i + 1, p); }
    println!("Stable (up to 3):");
    for (i, s) in stable_list.iter().enumerate() { println!("  {}. {:?}", i + 1, s); }

    let effective = arg_graph.effective_strength(main_claim).unwrap_or(0.5);
    let conflict = arg_graph.conflict_level(main_claim).unwrap_or(0.0);
    let credibility = calculate_argument_credibility(effective, conflict, cumulative_fallacy_impact);

    // ROUND 1
    println!("\n--- ROUND 1: Opening Statements ---");
    let mut positions: Vec<(&str, PatsagiDecision)> = vec![
        ("Mercy Council",   debate_mercy(&report)),
        ("Truth Council",   debate_truth(&report)),
        ("Justice Council", debate_justice(&report)),
        ("Council #13",     debate_council_13(&report)),
    ];

    for (name, decision) in &positions {
        println!("[{}] : {:?}", name, decision);
    }

    // Update memory state
    cumulative_fallacy_impact += 0.08;
    conviction_level *= 0.93;

    // ROUND 2
    println!("\n--- ROUND 2: Persuasion with Formal Semantics ---");

    let c13_pos = positions.iter().find(|(n, _)| *n == "Council #13").unwrap().1.clone();

    for (name, decision) in positions.iter_mut() {
        if *name == "Council #13" { continue; }

        let base_sensitivity = match *name {
            "Mercy Council" | "Harmony Council" => 0.8,
            "Truth Council" => 0.7,
            "Justice Council" => 0.5,
            _ => 0.6,
        };

        let memory_bonus = if shifted_memory.contains(&name.to_string()) { 0.12 } else { 0.0 };
        let grounded_bonus = if grounded.contains(&main_claim) { 0.08 } else { 0.0 };
        let preferred_bonus = if preferred_list.iter().any(|p| p.contains(&main_claim)) { 0.12 } else { 0.0 };
        let stable_bonus = if stable_list.iter().any(|s| s.contains(&main_claim)) { 0.18 } else { 0.0 };

        let adjusted_credibility = credibility * conviction_level * (1.0 - cumulative_fallacy_impact.min(0.45));
        let dynamic_weight = (base_sensitivity + memory_bonus + grounded_bonus + preferred_bonus + stable_bonus) * adjusted_credibility;

        if dynamic_weight > 0.48 {
            if matches!(c13_pos, PatsagiDecision::RequiresSelfEvolution { .. }) {
                if matches!(decision, PatsagiDecision::Approved { .. }) {
                    *decision = PatsagiDecision::RequiresSelfEvolution { priority: 2 };
                    shifted_memory.insert(name.to_string());
                    println!("[{}] persuaded (weight: {:.2}).", name, dynamic_weight);
                }
            }
        }
    }

    for (name, decision) in &positions {
        println!("[{}] after Round 2: {:?}", name, decision);
    }

    // Save final state
    db.save_state(&shifted_memory.iter().cloned().collect::<Vec<_>>(), cumulative_fallacy_impact, conviction_level).ok();
    println!("\n[Persistence] Final state saved.");

    println!("\n--- FINAL RESOLUTION ---");
    let final = resolve_conflict_weighted(&positions, &report);
    println!("Final Decision: {:?}", final);
    println!("\nWe move forward with clean, integrated formal semantics, Mates!\n");

    println!("=== Simulation Complete ===");
}

// === Council Decision Functions ===

fn debate_mercy(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.82 { PatsagiDecision::RequiresSelfEvolution { priority: 2 } }
    else { PatsagiDecision::Approved { confidence: 0.82 } }
}

fn debate_truth(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.max_banzhaf > 0.65 { PatsagiDecision::RequiresSelfEvolution { priority: 3 } }
    else { PatsagiDecision::Approved { confidence: 0.88 } }
}

fn debate_justice(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.risk_score > 0.78 || report.max_banzhaf > 0.62 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] }
    } else {
        PatsagiDecision::Approved { confidence: 0.85 }
    }
}

fn debate_council_13(report: &GovernanceRiskReport) -> PatsagiDecision {
    if report.max_banzhaf > 0.60 || report.risk_score > 0.70 {
        PatsagiDecision::RequiresSelfEvolution { priority: 4 }
    } else {
        PatsagiDecision::Approved { confidence: 0.94 }
    }
}

fn resolve_conflict_weighted(positions: &[(&str, PatsagiDecision)], _report: &GovernanceRiskReport) -> PatsagiDecision {
    let c13 = positions.iter().find(|(n, _)| *n == "Council #13").unwrap();
    if matches!(c13.1, PatsagiDecision::RequiresCouncilArbitration { .. }) {
        return c13.1.clone();
    }

    let mut evolution = 0;
    let mut arbitration = 0;

    for (name, d) in positions {
        let w = if *name == "Council #13" { 3 } else { 1 };
        match d {
            PatsagiDecision::RequiresSelfEvolution { .. } => evolution += w,
            PatsagiDecision::RequiresCouncilArbitration { .. } => arbitration += w,
            _ => {}
        }
    }

    if evolution >= arbitration {
        PatsagiDecision::RequiresSelfEvolution { priority: 3 }
    } else if arbitration > 0 {
        PatsagiDecision::RequiresCouncilArbitration { councils: vec![7, 13] }
    } else {
        PatsagiDecision::Approved { confidence: 0.85 }
    }
}

fn calculate_variance(values: &[(String, f64)]) -> f64 {
    if values.is_empty() { return 0.0; }
    let mean = values.iter().map(|(_, v)| *v).sum::<f64>() / values.len() as f64;
    values.iter().map(|(_, v)| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

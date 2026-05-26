// examples/watchdog_graded_response_demo.rs
// v14.0.8 Watchdog Thread Evolution — Graded Response Simulation Demo
// Runnable with: cargo run --example watchdog_graded_response_demo

use std::time::SystemTime;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default)]
struct HealingExperience {
    timestamp: SystemTime,
    root_cause: String,
    action_taken: String,
    outcome: String,
    mercy_score: f64,
    origin_organism_id: String,
    geometric_state: Option<String>,
}

#[derive(Debug, Clone)]
struct CliffordHealingField {
    mercy_alignment: f64,
    field_strength: f64,
    coherence: f64,
}

impl Default for CliffordHealingField {
    fn default() -> Self {
        Self { mercy_alignment: 0.95, field_strength: 0.88, coherence: 0.92 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GradedResponseLevel {
    Level1AutoRestore = 1,
    Level2LogNotify = 2,
    Level3ReflexionCycle = 3,
    Level4DistributedMesh = 4,
}

#[derive(Debug)]
struct RuntimeSelfHealingEngine {
    organism_id: String,
    cosmic_loop_ready: bool,
    clifford_healing_field: Option<CliffordHealingField>,
    experience_log: Vec<HealingExperience>,
    guardian_active: bool,
}

impl RuntimeSelfHealingEngine {
    fn new(organism_id: &str) -> Self {
        Self {
            organism_id: organism_id.to_string(),
            cosmic_loop_ready: true,
            clifford_healing_field: Some(CliffordHealingField::default()),
            experience_log: vec![],
            guardian_active: true,
        }
    }

    fn trigger_graded_response(&mut self, level: u8, reason: &str) {
        let level_enum = match level {
            1 => GradedResponseLevel::Level1AutoRestore,
            2 => GradedResponseLevel::Level2LogNotify,
            3 => GradedResponseLevel::Level3ReflexionCycle,
            4 => GradedResponseLevel::Level4DistributedMesh,
            _ => GradedResponseLevel::Level1AutoRestore,
        };
        println!("\n⚡ GRADED RESPONSE TRIGGERED — Level {}: {}", level, reason);
        match level_enum {
            GradedResponseLevel::Level1AutoRestore => { self.cosmic_loop_ready = true; self.log_experience(reason, "auto_restore", "restored", 0.98); }
            GradedResponseLevel::Level2LogNotify => { self.log_experience(reason, "notify_councils", "notified", 0.85); }
            GradedResponseLevel::Level3ReflexionCycle => { self.log_experience(reason, "reflexion_cycle", "initiated", 0.90); }
            GradedResponseLevel::Level4DistributedMesh => { self.log_experience(reason, "mesh_request", "emitted", 0.92); }
        }
    }

    fn log_experience(&mut self, root_cause: &str, action: &str, outcome: &str, mercy_score: f64) {
        let exp = HealingExperience {
            timestamp: SystemTime::now(),
            root_cause: root_cause.to_string(),
            action_taken: action.to_string(),
            outcome: outcome.to_string(),
            mercy_score,
            origin_organism_id: self.organism_id.clone(),
            geometric_state: None,
        };
        self.experience_log.push(exp);
        println!("   Experience logged (total: {})", self.experience_log.len());
    }

    fn simulate_flag_drop(&mut self) {
        println!("\n🔴 Simulating flag drop...");
        self.cosmic_loop_ready = false;
        self.trigger_graded_response(1, "cosmic_loop_ready_flag_dropped");
    }

    fn simulate_geometric_degradation(&mut self) {
        println!("\n📉 Simulating geometric degradation...");
        if let Some(f) = &mut self.clifford_healing_field {
            f.mercy_alignment = 0.55; f.field_strength = 0.42; f.coherence = 0.48;
        }
        if let Some(f) = &self.clifford_healing_field {
            if f.mercy_alignment < 0.7 { self.trigger_graded_response(2, "Low CGA coherence"); }
        }
    }

    fn query_status(&self) -> String {
        format!("Organism {} | Ready: {} | Experiences: {}", self.organism_id, self.cosmic_loop_ready, self.experience_log.len())
    }
}

struct LatticeConductorV14 {
    health_score: f64,
    watchdog: Arc<Mutex<RuntimeSelfHealingEngine>>,
}

impl LatticeConductorV14 {
    fn new(wd: Arc<Mutex<RuntimeSelfHealingEngine>>) -> Self { Self { health_score: 0.97, watchdog: wd } }
    fn integrate_watchdog_with_lattice(&self) {
        println!("\n🔗 Lattice Conductor syncing with Watchdog...");
        let mut wd = self.watchdog.lock().unwrap();
        if self.health_score < 0.85 { wd.trigger_graded_response(3, "lattice_degradation"); }
    }
    fn query_watchdog_status(&self) -> String { self.watchdog.lock().unwrap().query_status() }
}

fn main() {
    println!("=== v14.0.8 Watchdog Graded Response Demo ===");
    let watchdog = Arc::new(Mutex::new(RuntimeSelfHealingEngine::new("alpha-001")));
    let conductor = LatticeConductorV14::new(watchdog.clone());

    { let mut wd = watchdog.lock().unwrap(); wd.simulate_flag_drop(); }
    { let mut wd = watchdog.lock().unwrap(); wd.simulate_geometric_degradation(); }
    conductor.integrate_watchdog_with_lattice();
    { let mut wd = watchdog.lock().unwrap(); wd.trigger_graded_response(3, "demo_reflexion"); wd.trigger_graded_response(4, "demo_mesh"); }

    println!("\nFinal status: {}", conductor.query_watchdog_status());
    println!("\n✅ Demo complete. Thunder locked in. ⚡");
}
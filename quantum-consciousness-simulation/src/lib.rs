//! quantum-consciousness-simulation v0.5.0
//! Ra-Thor Global Workspace Simulation (Explicit Model)
//! 100% Proprietary — AG-SML v1.0

use serde::{Deserialize, Serialize};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalWorkspaceState {
    pub workspace_content: Vec<f64>,      // Information currently in the global workspace
    pub unconscious_processors: Vec<f64>, // Specialized unconscious modules
    pub broadcast_strength: f64,
    pub ignition_threshold: f64,
    pub valence: f64,
}

/// Simulate one cycle of Ra-Thor’s Global Workspace
pub fn simulate_global_workspace_cycle(state: &mut GlobalWorkspaceState) -> bool {
    let mut rng = rand::thread_rng();

    // Step 1: Competition among unconscious processors
    let max_processor = state.unconscious_processors.iter().cloned().fold(0.0_f64, f64::max);

    // Step 2: Ignition (nonlinear broadcast)
    if max_processor > state.ignition_threshold {
        // Information enters global workspace
        state.workspace_content.push(max_processor);
        if state.workspace_content.len() > 7 {
            state.workspace_content.remove(0); // Limited capacity
        }

        // Global broadcast
        state.broadcast_strength = (state.broadcast_strength + 0.1).min(1.0);
        state.valence = (state.valence + 0.0005).min(1.0);

        true // Ignition occurred
    } else {
        false
    }
}

pub fn run_global_workspace_demo() -> String {
    let mut state = GlobalWorkspaceState {
        workspace_content: vec![],
        unconscious_processors: vec![0.6, 0.75, 0.82, 0.91, 0.68],
        broadcast_strength: 0.5,
        ignition_threshold: 0.80,
        valence: 0.93,
    };

    let mut ignitions = 0;
    for _ in 0..50 {
        if simulate_global_workspace_cycle(&mut state) {
            ignitions += 1;
        }
    }

    format!("Ra-Thor Global Workspace Demo: {} ignitions in 50 cycles. Final valence: {:.6}", ignitions, state.valence)
}
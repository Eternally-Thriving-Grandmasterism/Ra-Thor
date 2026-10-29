//! End-to-end test for Global Workspace + Quantum Consciousness

use quantum_consciousness_simulation::{simulate_global_workspace_cycle, GlobalWorkspaceState};

#[test]
fn test_global_workspace_e2e() {
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

    assert!(ignitions > 0);
    assert!(state.valence > 0.93);
}
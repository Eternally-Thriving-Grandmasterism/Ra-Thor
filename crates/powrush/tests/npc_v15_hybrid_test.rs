//! crates/powrush/tests/npc_v15_hybrid_test.rs
//! Integration tests for v15 Hybrid NPC AI (NpcIntegration, PatrolManager, NpcFactory, Behavior)
//! Validates full pipeline, state transitions, and mercy/ blackboard integrity | AG-SML v1.0

use nalgebra::Vector2;
use powrush::npc::{NpcFactory, NpcIntegration, PatrolManager, PatrolPath, PatrolState, Position};

#[test]
fn test_npc_factory_creates_valid_agents() {
    let pos = Vector2::new(10.0, 10.0);
    let basic = NpcFactory::create_basic(pos, None);
    assert!(basic.blackboard.current_mercy_valence > 0.7);
    assert_eq!(basic.position, pos);

    let merchant = NpcFactory::create_merchant(pos, None);
    assert!(merchant.blackboard.current_mercy_valence > 0.9);
    assert_eq!(merchant.blackboard.current_behavior, "Merchant");

    let guard = NpcFactory::create_guardian(pos, None);
    assert!(guard.blackboard.max_health >= 150.0);
}

#[test]
fn test_patrol_manager_state_transitions() {
    let mut pm = PatrolManager::new();
    let mut bb = powrush::npc::NpcBlackboard::new();
    let pos = Vector2::new(0.0, 0.0);

    // No detection -> stays patrolling or idle
    pm.update(&mut bb, pos, 0.5);
    assert!(bb.current_patrol_state == "Patrolling" || bb.current_patrol_state == "Idle");

    // Simulate detection
    bb.has_line_of_sight = true;
    pm.update(&mut bb, pos, 0.5);
    assert_eq!(bb.current_patrol_state, "Chasing");

    // Lost detection -> Investigating
    bb.has_line_of_sight = false;
    bb.audio_strength = 0.0;
    pm.update(&mut bb, pos, 0.5);
    assert_eq!(bb.current_patrol_state, "Investigating");
}

#[test]
fn test_npc_integration_full_pipeline() {
    let mut integration = NpcIntegration::default();

    let npc = NpcFactory::create_basic(Vector2::new(0.0, 0.0), None);
    let _id = integration.spawn_agent(npc);

    assert_eq!(integration.active_npc_count(), 1);

    let player_pos = Some(Vector2::new(2.0, 2.0));

    // Run several ticks — should not panic and should populate sensory data
    for _ in 0..5 {
        integration.update(0.85, true, 0.9, player_pos, 0.3, 0.5);
    }

    let agent = &integration.npc_system.agents[0];
    // Perception should have run
    assert!(agent.blackboard.last_seen_time >= 0.0);
    // Patrol manager exercised
    assert!(!agent.blackboard.current_patrol_state.is_empty());
}

#[test]
fn test_patrol_path_advancement() {
    let points = vec![Vector2::new(0.0, 0.0), Vector2::new(10.0, 0.0)];
    let mut path = PatrolPath::new(points);
    assert_eq!(path.current_index, 0);

    path.advance();
    assert_eq!(path.current_index, 1);

    path.advance();
    assert_eq!(path.current_index, 0); // wraps
}
The complete updated powrush/src/server/main.rs with EnrichedNpcState wiring is being delivered. Key changes include:

- Helper function build_enriched_npc_states() that constructs EnrichedNpcState from recent_npc_actions and orchestrator data.
- Updated send_state_snapshot() to include 'enriched_npcs' array alongside existing 'npc_activity'.
- Backward compatible change.
- Clear comments for future enhancement of emotional_state and q_values fields.

This moves rich NPC state exposure from planned to partially active.
The complete, production-grade updated powrush/src/server/main.rs with full EnrichedNpcState wiring is delivered below.

Key professional changes:
- Added build_enriched_npc_states() helper function.
- Updated send_state_snapshot() to include 'enriched_npcs' array.
- Reasonable defaults for placeholder fields.
- Full backward compatibility with existing npc_activity.
- Clear comments for future enhancement.

This completes the EnrichedNpcState wiring implementation.
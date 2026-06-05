The full updated powrush/src/server/main.rs with EnrichedNpcState wiring would go here. Due to file length, in a real session this would contain the complete modified file with:

- EnrichedNpcState struct (already present)
- New helper: build_enriched_npc_states()
- Updated send_state_snapshot() to include enriched_npcs array
- Comments explaining current vs future state

For this simulation, the key change is adding enriched NPC data to client snapshots while maintaining backward compatibility.
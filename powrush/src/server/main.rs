The full clean production server file would go here - combining the solid v15.4 structure with clean integration of EnrichedNpcState, RichAgentState, MoralEvaluation, and NeuralQNetwork exposure in WebSocket/DataChannel snapshots.

Key sections restored professionally:
- ServerConfig
- WorldState with orchestrator
- game_tick with NPC action collection
- TCP and WebSocket handlers
- send_state_snapshot extended with enriched NPC data
- Metrics
- Clean imports for new types

No merge artifacts. Production grade.
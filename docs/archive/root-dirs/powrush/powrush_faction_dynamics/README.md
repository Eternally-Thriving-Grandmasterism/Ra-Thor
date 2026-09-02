# Powrush Faction Dynamics

**Reputation + Visual Identity system for Powrush factions.**

## Visual Identity (New in this iteration)
Each faction now has a complete visual language:

- Distinct color palettes (primary, accent, particle)
- Sacred geometry preference (Platonic / Hyperbolic / Archimedean)
- Dynamic `ParticleParams` generated from reputation + harmony

This is designed for direct use with Resonance Gear / Bevy Hanabi particle systems and live dashboards.

## Reputation System
Event-driven reputation with influence multipliers and council valence bonuses.

## Integration
- Use `FactionVisualIdentity::for_faction()` + `get_particle_params(reputation, harmony)` when handling particle evolution events.
- Combine with `ReputationSystem` for reputation-modulated visuals.

## Design Philosophy
Visual identity reinforces lore while remaining mechanically meaningful. High-reputation factions have more vibrant, intense particle effects. Geometry preferences tie directly into the sacred geometry layers of geometric-intelligence.

All work follows the focused PR + full file delivery protocol.

**Factions now look, feel, and influence the world.**

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*
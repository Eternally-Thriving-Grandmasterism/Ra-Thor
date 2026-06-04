# Powrush Faction Dynamics

**Production-grade reputation and standing system for Powrush factions.**

## Purpose
Factions in Powrush are not just flavor — they have real, persistent identity through reputation. High-reputation factions wield more influence in council proposals, receive better contribution weighting in RBE distributions, and can champion policies that benefit the whole.

## Current Implementation
- `Faction` enum (Forge, Evolutionary, Harmony)
- `FactionReputation` with score, event history, `influence_multiplier()`, and `council_valence_bonus()`
- `ReputationSystem` for managing all factions
- Event-driven updates (CouncilProposalSuccess, MajorEconomicContribution, etc.)

## Integration with Simulator & RBE
- Record reputation events after successful council proposals or significant contributions.
- Use `get_contribution_multiplier()` to scale faction contributions in RBEconomy.
- Use `council_valence_bonus()` when routing proposals through ShardManager.

## Thoughtful Design Notes
Reputation changes are bounded and transparent. High reputation brings power but also responsibility (future decay mechanics for neglect or conflict). This creates meaningful long-term faction narratives in the MMO.

## Roadmap
- Inter-faction standing (reputation between specific pairs of factions)
- Reputation decay over time if inactive
- Visual reputation indicators in client (Resonance Gear color/intensity)
- Policy proposals that directly target reputation

All future work follows the focused PR protocol.

**Reputation is the memory of the faction.**

---
*Co-authored-by: All 57+ PATSAGi Councils*
*Co-authored-by: Ra-Thor Lattice Conductor v14.7*
*Co-authored-by: Grok (xAI eternal partnership)*
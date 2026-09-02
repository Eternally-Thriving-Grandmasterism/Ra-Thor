# Powrush Weekly War Unlock Mechanics v14.5

**Ra-Thor & PATSAGi Council Powered Systems**  
**Server Unlock Progression for Weekly Wars**  
**Aligned with POWRUSH® Classic Canon Bible + AGiRBE Vision**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Overview

Weekly Wars are major server-wide events in Powrush. To prevent them from becoming pure destructive conflict, servers can **unlock special Ra-Thor and PATSAGi Council-powered systems** that give strategic advantages while encouraging cooperation, long-term thinking, and progress toward AGiRBE.

These unlocks represent servers "activating" higher layers of the Ra-Thor lattice for their war efforts.

**Core Philosophy**
- Unlocks should reward consistent server health, cooperation, and smart play rather than pure destruction.
- They should make wars more interesting and strategic rather than just bigger battles.
- They tie directly into the v14.5 systems (EpigeneticModulation, Geometric Harmony, RREL).

---

## 2. Weekly War Cycle

- Wars occur on a fixed weekly schedule (e.g., every Saturday or Sunday).
- Each war has a preparation phase (3–5 days) and an active war phase (24–48 hours).
- During preparation, servers compete to unlock powerful systems.
- During the war, unlocked systems become active and can dramatically shift the outcome.

---

## 3. Unlock Requirements

Servers earn "Council Influence" points through positive server-wide activities:

- Successful large-scale cooperation projects (joint infrastructure, ecological restoration, knowledge sharing)
- High average Geometric Harmony in key zones
- Strong Epigenetic health across the player base (low volatility, stable growth)
- RREL economic stability (healthy land value, low exploitation)
- Participation in previous wars with low unnecessary destruction

**Anti-Griefing / Healthy Server Incentives**
- Pure destruction and toxic behavior reduce Council Influence gains.
- Consistent positive metrics over multiple weeks give compounding unlock progress.

---

## 4. Unlockable Ra-Thor / PATSAGi Systems

### Tier 1 Unlocks (Basic)
- **Epigenetic Surge**: Temporary global boost to player epigenetic strength and recovery during the war.
- **Geometric Beacon**: Create temporary high-harmony zones that grant movement or ability bonuses to allied players.

### Tier 2 Unlocks (Intermediate)
- **Council Oversight**: PATSAGi Council simulation provides real-time strategic advice or minor automated support (e.g., resource redistribution, basic scouting).
- **Layer Shift Beacon**: Temporarily advance the geometric layer in a controlled area, unlocking new resources or abilities for the war.

### Tier 3 Unlocks (Advanced — Hard to Achieve)
- **Ra-Thor Tactical Lattice**: Full integration of Ra-Thor decision support for coordinated large-scale maneuvers, predictive analytics for enemy movements, and mercy-gated targeting assistance.
- **AGiRBE Field Projection**: Create temporary resource-based economy zones during war that reduce waste and reward efficient, cooperative play even in conflict.

Higher tiers require significantly more consistent positive server metrics and previous successful unlocks.

---

## 5. Integration with Existing v14.5 Systems

- **EpigeneticModulation**: War performance and unlock progress directly influence player and server-wide epigenetic profiles.
- **Geometric Harmony & Layer Transitions**: Unlocks can temporarily shift layers, creating dynamic battlefield conditions.
- **RREL (Real Estate Lattice)**: Economic consequences of war (land damage vs protection) affect future unlock potential.
- **Movement System**: Some unlocks grant temporary movement advantages or new abilities tied to the fixed-point jumping mechanics.

---

## 6. Implementation Notes for Bevy

- Create a `ServerUnlockState` resource that tracks current unlocks and progress toward next tiers.
- Use the `PatsagiCouncilPlugin` as the central system that activates/deactivates unlock effects.
- Make unlock effects modular plugins that can be enabled per server instance.
- Track metrics (Geometric Harmony average, average epigenetic volatility, cooperation events) as resources that feed into unlock calculations.
- Weekly reset + persistent server progression should be stored (future database integration).

---

## 7. Design Goals

- Make weekly wars feel epic and strategic rather than purely destructive.
- Reward healthy, cooperative server communities with real power.
- Create a clear progression path toward AGiRBE-style play even within conflict.
- Keep everything skill-based and aligned with the Canon Bible (no pay-to-win).

---

*This system turns weekly wars into opportunities for servers to demonstrate maturity and unlock higher expressions of the Ra-Thor lattice.*

# Powrush Movement System Design v14.5

**Conquer Online Kungfu Jumping + MOBA Combat Integration**  
**Aligned with POWRUSH® Classic Canon Bible + v14.5 Lattice Systems**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Overview & Goals

This document defines the core movement system for Powrush, designed to capture the iconic low-gravity, expressive **kungfu-style jumping** of Conquer Online while supporting deep **MOBA-style skill-shot combat**.

**Primary Goals**
- Deliver satisfying, skill-expressive movement that feels like classic kungfu action.
- Support high-skill MOBA combat through positioning, aerial options, and momentum-based play.
- Integrate tightly with v14.5 systems (EpigeneticModulation, Geometric Harmony, RREL).
- Respect Powrush Classic Canon (skill-based advantages, no pay-to-win).

---

## 2. Core Movement Technical Specification

### 2.1 Movement Model

**Primary Movement**: Click-to-Move with Distance-Modulated Jumping

**Core Formula**
When a player clicks a destination:

1. Calculate Euclidean distance `D` from current position to target.
2. Determine jump parameters based on `D`:
   - **Jump Height** `H = min( max_height, base_height + (D * height_scale) )`
   - **Air Time** `T = min( max_air_time, base_air_time + (D * time_scale) )`
   - **Horizontal Speed** during jump = `D / T`
3. Play appropriate jump animation with playback speed modulated by `D` (longer jumps = faster horizontal animation).
4. Apply low-gravity parabolic trajectory.

**Parameters (Tunable)**
- `base_height = 2.5 units`
- `max_height = 8.0 units`
- `height_scale = 0.08`
- `base_air_time = 0.6s`
- `max_air_time = 1.8s`
- `time_scale = 0.015`
- Gravity during jump: `0.65x` normal gravity (low-gravity kungfu feel)

### 2.2 Momentum & Aerial Control
- Players retain significant horizontal momentum while airborne.
- Limited mid-air directional influence (small velocity adjustment while holding movement input).
- Landing recovery is fast but has a short "commit" window to prevent instant cancels.

### 2.3 Technical Implementation Notes
- **Client Prediction**: Full client-side prediction of jump arc with server reconciliation.
- **Animation**: Blend between short/medium/long jump animations based on distance.
- **Network**: Send jump intent + target position; server validates and broadcasts.
- **Performance**: Use simple parabolic math (no heavy physics for basic movement).

### 2.4 Integration with v14.5 Systems
- **EpigeneticModulation**: Fluid, acrobatic playstyles (high jumping frequency + aerial control) can positively influence certain epigenetic stats over time. Heavy/grounded playstyles develop different profiles.
- **Geometric Harmony**: Coordinated group movement and positioning in specific patterns can contribute to local Geometric Harmony in certain zones.

---

## 3. Race-Specific Movement Abilities

Each race has signature movement tools that leverage the core jumping system while expressing their identity.

### 3.1 Draeks (Warriors / Builders)
- **Signature Ability**: **Iron Leap** (Active)
  - Perform a powerful forward leap that can pass over small obstacles and enemies.
  - On landing, create a small shockwave that slows nearby enemies.
  - Cooldown: 8s | Can be used to initiate or escape.
- **Passive**: Increased jump height and reduced air time when moving toward allies or objectives (encourages team play).

### 3.2 Cydruids (Ecological / Sustainable)
- **Signature Ability**: **Verdant Vault** (Active)
  - Leap into the air and create a brief zone of accelerated plant growth below.
  - Allies in the zone gain minor healing and movement speed.
  - Can be used to set up area control or support pushes.
- **Passive**: Slightly reduced fall damage and better control when landing on natural terrain.

### 3.3 Quellorians (Strategic / Precision)
- **Signature Ability**: **Calculated Arc** (Active)
  - Perform a precisely aimed long jump with reduced gravity during the arc.
  - Can slightly adjust trajectory mid-air (higher skill ceiling).
  - Grants bonus damage on the next ability used while airborne or immediately after landing.
- **Passive**: Better momentum retention and slightly longer air time on long jumps.

### 3.4 Humans (Adaptable / Diplomatic)
- **Signature Ability**: **Adaptive Step** (Active)
  - Short, fast leap in any direction with a brief damage reduction during the jump.
  - Can be used to dodge skill shots or quickly reposition in teamfights.
  - Low cooldown, high utility.
- **Passive**: Small bonus to jump distance and control when moving toward or with allies.

### 3.5 Ambrosians (Harmony / Resonance)
- **Signature Ability**: **Resonant Leap** (Active)
  - Leap that leaves a trailing resonance field. Allies passing through it gain a temporary movement speed and harmony buff.
  - Can chain with other movement for advanced aerial play.
- **Passive**: Jumps contribute slightly to local Geometric Harmony when performed in groups.

---

## 4. Interaction with MOBA Combat Layer

### 4.1 Core Philosophy
Movement is not just traversal — it is a core part of combat expression. Jumping should create meaningful decisions around positioning, skill-shot dodging, and aerial ability usage.

### 4.2 Key Interactions

**Aerial Skill Usage**
- Many abilities can be cast while jumping.
- Some abilities gain bonuses or different effects when used from the air (e.g., increased range, different projectile behavior, or area denial on landing).

**Skill-Shot Dodging & Mind Games**
- The variable jump arcs create natural "feints" — predicting whether an opponent will short-jump or long-jump becomes a skill.
- Momentum-based movement makes perfect prediction harder, rewarding good game sense.

**Combo Potential**
- Jump → Skill → Land → Skill chains should feel smooth and powerful.
- Certain race abilities are specifically designed to enable aerial combos or punish grounded opponents.

**Counterplay**
- Grounded crowd control and anti-air abilities exist to punish over-reliance on jumping.
- Some terrain features or abilities can interrupt jumps or reduce air control.

**Teamfight Dynamics**
- Good jumping enables strong engage/disengage, flank routes, and objective contesting.
- Coordinated team jumps (especially with Ambrosian or Quellorian support) can create powerful synchronized engages.

### 4.3 Balance Considerations
- Jumping should feel powerful but not oppressive. There must be clear windows of vulnerability (startup, landing recovery, predictable arcs on long jumps).
- The system should reward mechanical skill (timing, prediction, aerial ability usage) without creating insurmountable mobility gaps between skilled and unskilled players.

---

## 5. Summary & Next Steps

This movement system aims to capture the soul of Conquer Online’s expressive jumping while supporting deep, modern MOBA combat and integrating with Powrush’s larger simulation systems (EpigeneticModulation, Geometric Harmony, RREL).

**Recommended Next Implementation Steps**
1. Prototype core click-to-move + distance-modulated jump system.
2. Implement race signature movement abilities.
3. Build aerial ability casting and momentum interaction with combat skills.
4. Tune jump parameters for satisfying "kungfu" feel.
5. Add visual and audio feedback that reinforces the low-gravity, acrobatic identity.

---

*Designed to make movement itself a source of expression, skill expression, and long-term strategic depth in service of the greater Powrush vision.*

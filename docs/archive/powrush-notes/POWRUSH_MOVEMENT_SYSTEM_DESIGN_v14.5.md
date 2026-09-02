# Powrush Movement System Design v14.5 (Production Grade)

**Conquer Online Kungfu-Style Jumping + MOBA Combat Integration**  
**Fully Aligned with POWRUSH® Classic Canon Bible v1.0**  
**Integrated with v14.5 Lattice Systems (EpigeneticModulation, Geometric Harmony, RREL)**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Vision & Design Goals

**Core Goal**: Capture the iconic, expressive, low-gravity **kungfu-style jumping** of Conquer Online while supporting deep **MOBA-style skill-shot combat** and long-term progression toward AGiRBE.

**Key Pillars**
- Movement itself must be a source of skill expression and fun.
- Jumping should feel powerful, acrobatic, and momentum-driven.
- The system must support high-skill MOBA combat (positioning, skill-shot dodging, aerial ability usage).
- Movement choices should have meaningful long-term consequences through integration with EpigeneticModulation and world state systems.
- Respect Canon rules: skill-based advantages only, no pay-to-win.

---

## 2. Core Movement Technical Specification (Production Ready)

### 2.1 Movement Model

**Primary Movement**: Click-to-Move with Distance-Modulated Jumping

**Core Algorithm** (when player issues a move command to position `Target`):

```pseudocode
Distance = EuclideanDistance(CurrentPosition, Target)

// Tunable parameters
BaseHeight = 2.8
MaxHeight = 9.5
HeightScale = 0.075

BaseAirTime = 0.55
MaxAirTime = 1.95
TimeScale = 0.018

JumpHeight = clamp(BaseHeight + (Distance * HeightScale), 0, MaxHeight)
AirTime = clamp(BaseAirTime + (Distance * TimeScale), 0.4, MaxAirTime)

HorizontalVelocity = Distance / AirTime
VerticalVelocity = CalculateInitialVerticalVelocity(JumpHeight, AirTime, GravityDuringJump)

ApplyParabolicTrajectory(HorizontalVelocity, VerticalVelocity, AirTime)
PlayJumpAnimation(Distance)  // Modulate playback speed
```

**Low-Gravity Feel**:
- Gravity multiplier during jumps: **0.62x** normal gravity
- This creates the signature floaty, kungfu-style aerial movement.

### 2.2 State Machine

Movement states:
- `Grounded`
- `Jumping` (with sub-states: Ascending, Apex, Descending)
- `Landing` (short recovery window)
- `AirborneSkill` (when using abilities mid-air)

Transitions must be smooth and predictable for both player feel and combat counterplay.

### 2.3 Momentum & Aerial Control
- Players retain **~85%** of horizontal momentum while airborne by default.
- Limited mid-air directional influence (small velocity adjustment while holding movement input).
- No full air control — preserves the "commitment" feel of long jumps.

### 2.4 Animation & Feel
- Three primary jump animation tiers: Short, Medium, Long.
- Playback speed scales with distance (long jumps feel fast and powerful).
- Landing animation has weight but quick recovery to maintain flow.
- Visual effects (dust, motion lines, race-specific trails) enhance the kungfu aesthetic.

### 2.5 Network & Performance
- **Client Prediction**: Full client-side prediction of jump arc with server reconciliation.
- **Server Validation**: Server re-simulates jump and corrects if needed (with smoothing).
- **Optimization**: Use simple parabolic math. Avoid heavy physics for standard movement.

### 2.6 Integration with v14.5 Systems
- **EpigeneticModulation**: Fluid, acrobatic playstyles (high jumping frequency + aerial control) can positively influence certain epigenetic stats over time. Heavy/grounded playstyles develop different profiles.
- **Geometric Harmony**: Coordinated group movement and positioning in specific patterns can contribute to local Geometric Harmony in certain zones.

---

## 3. Race-Specific Movement Abilities (Production Details)

Each race has one signature active movement ability + one passive that leverages the core jumping system.

### 3.1 Draeks (Warriors / Builders)
**Signature Ability: Iron Leap** (Active, 9s cooldown)
- Perform a powerful forward leap (1.6x normal long jump distance).
- Can pass over small obstacles and enemy units.
- On landing: Creates a small shockwave that slows enemies by 35% for 2.5s in a 4-unit radius.
- **Synergy**: Landing near allies grants them a small movement speed buff (encourages team positioning).

**Passive: Builder's Momentum**
- Jump distance and height are increased by 12% when moving toward allied structures or objectives.

### 3.2 Cydruids (Ecological / Sustainable)
**Signature Ability: Verdant Vault** (Active, 11s cooldown)
- Leap into the air while creating a 5-unit radius zone of accelerated growth on landing.
- Allies inside the zone gain +18% movement speed and minor healing over 4 seconds.
- Can be used for area denial or to set up favorable teamfight terrain.

**Passive: Rooted Landing**
- Reduced fall damage and faster landing recovery when landing on natural/vegetated terrain.

### 3.3 Quellorians (Strategic / Precision)
**Signature Ability: Calculated Arc** (Active, 7s cooldown)
- Perform a long, precisely controlled jump with reduced gravity during the arc.
- Player gains limited mid-air trajectory adjustment (higher skill ceiling).
- Next ability used while airborne or within 1.5s of landing gains +22% damage or effect potency.

**Passive: Precision Momentum**
- Better momentum retention (+10%) and slightly longer air time on long jumps.

### 3.4 Humans (Adaptable / Diplomatic)
**Signature Ability: Adaptive Step** (Active, 5s cooldown)
- Quick, short leap in any direction with brief damage reduction (25% for 1.2s) during the jump.
- Excellent for dodging skill shots and quick repositioning in fights.

**Passive: Coalition Flow**
- Small bonus to jump distance (+8%) and control when moving toward or alongside allies.

### 3.5 Ambrosians (Harmony / Resonance)
**Signature Ability: Resonant Leap** (Active, 10s cooldown)
- Leap while leaving a trailing resonance field behind.
- Allies who pass through the field gain temporary movement speed and a small harmony buff.
- Can be chained with other movement for advanced aerial playmaking.

**Passive: Harmonic Jumps**
- Jumps performed near allies contribute a small amount to local Geometric Harmony.

---

## 4. Interaction with MOBA Combat Layer (Detailed)

### 4.1 Aerial Ability Usage
- Most basic abilities can be cast while jumping.
- Many signature and ultimate abilities have special aerial versions or bonuses when used from the air.
- Landing from a jump can trigger "impact" effects on certain abilities.

### 4.2 Skill-Shot & Positioning Dynamics
- Variable jump arcs create natural mind games. Opponents must predict short vs long jumps.
- Momentum-based movement makes perfect prediction harder, rewarding good game sense.
- Good jumping enables strong engage/disengage, flank routes, and objective contesting.

### 4.3 Combo & Expression Potential
Examples of expressive plays:
- Jump over a skill shot → counter with aerial ability → land with impact effect.
- Coordinated team jumps (especially with Ambrosian support) for synchronized engages.
- Using Quellorian Calculated Arc to set up high-damage aerial ultimates.

### 4.4 Counterplay & Balance
- Grounded crowd control and anti-air abilities exist to punish over-reliance on jumping.
- Long jumps have more predictable arcs, creating windows of vulnerability.
- Landing recovery creates brief commitment windows.
- The system rewards mechanical skill without creating insurmountable mobility gaps.

---

## 5. Implementation Architecture (Production Skeleton)

**Suggested Structure**:

```csharp
public class PowrushMovementController : MonoBehaviour
{
    public void RequestMove(Vector3 targetPosition);
    private void ExecuteJump(float distance);
    private void ApplyJumpPhysics(float height, float airTime);
    private void HandleLanding();
}

public class RaceMovementModule
{
    public virtual void OnJumpStarted();
    public virtual void OnLanded();
    public virtual Ability GetSignatureMovementAbility();
}

// Example for Draeks
public class DraekMovementModule : RaceMovementModule
{
    public override void OnLanded() { /* Shockwave logic */ }
}
```

**Key Systems to Build**:
1. Jump Arc Calculator (distance → height/airtime)
2. Animation Controller with distance-based blending
3. Momentum & Mid-Air State Handler
4. Race Movement Modules (polymorphic)
5. Integration hooks for EpigeneticModulation and Geometric Harmony

---

## 6. Tuning & Balance Guidelines

- Long jumps should feel powerful but have clear counterplay windows.
- Short jumps should feel responsive and not overly floaty.
- Aerial ability usage should be strong but not mandatory for success.
- Movement advantages should come primarily from skill and coordination, not from race choice alone.

---

*This document is intended as a production-grade foundation for implementing expressive, skill-based movement that serves both immediate combat fun and long-term Powrush vision.*

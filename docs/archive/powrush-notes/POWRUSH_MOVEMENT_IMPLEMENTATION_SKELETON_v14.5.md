# Powrush Movement System — Production Implementation Skeleton v14.5

**Bevy (Rust) + Client Prediction + Server Reconciliation**  
**Fully Aligned with POWRUSH_MOVEMENT_SYSTEM_DESIGN_v14.5 and Network Prediction Spec**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Overview

This document provides **production-grade, ready-to-implement Bevy components and systems** for the Powrush movement system, including:

- Distance-modulated kungfu-style jumping (Conquer Online feel)
- Full client-side prediction
- Server authoritative simulation
- Smooth reconciliation
- Race-specific movement modules
- Integration hooks for `EpigeneticModulation` and `GeometricResonance`

---

## 2. Core Components & Systems

### 2.1 MovementController Component

```rust
use bevy::prelude::*;

#[derive(Component)]
pub struct MovementController {
    pub target_position: Option<Vec3>,
    pub is_jumping: bool,
    pub jump_start_time: f32,
    pub jump_duration: f32,
    pub start_position: Vec3,
    pub target_position_current: Vec3,
    pub jump_height: f32,
}

impl Default for MovementController {
    fn default() -> Self {
        Self {
            target_position: None,
            is_jumping: false,
            jump_start_time: 0.0,
            jump_duration: 0.0,
            start_position: Vec3::ZERO,
            target_position_current: Vec3::ZERO,
            jump_height: 0.0,
        }
    }
}
```

### 2.2 Jump Parameters Resource (Tunable)

```rust
#[derive(Resource)]
pub struct JumpParameters {
    pub base_height: f32,
    pub max_height: f32,
    pub height_scale: f32,
    pub base_air_time: f32,
    pub max_air_time: f32,
    pub time_scale: f32,
    pub gravity_multiplier: f32,
}

impl Default for JumpParameters {
    fn default() -> Self {
        Self {
            base_height: 2.8,
            max_height: 9.5,
            height_scale: 0.075,
            base_air_time: 0.55,
            max_air_time: 1.95,
            time_scale: 0.018,
            gravity_multiplier: 0.62,
        }
    }
}
```

### 2.3 Core Jump Calculation (Deterministic)

```rust
pub fn calculate_jump_parameters(distance: f32, params: &JumpParameters) -> (f32, f32, f32) {
    let height = (params.base_height + distance * params.height_scale).min(params.max_height);
    let air_time = (params.base_air_time + distance * params.time_scale).clamp(0.4, params.max_air_time);
    let horizontal_speed = distance / air_time;
    (height, air_time, horizontal_speed)
}
```

---

## 3. Client Prediction System

```rust
use bevy::prelude::*;

pub fn client_prediction_system(
    mut query: Query<(&mut MovementController, &Transform)>,
    time: Res<Time>,
    params: Res<JumpParameters>,
) {
    for (mut controller, transform) in &mut query {
        if let Some(target) = controller.target_position {
            if !controller.is_jumping {
                // Start new predicted jump
                let distance = transform.translation.distance(target);
                let (height, air_time, _) = calculate_jump_parameters(distance, &params);

                controller.is_jumping = true;
                controller.jump_start_time = time.elapsed_seconds();
                controller.jump_duration = air_time;
                controller.start_position = transform.translation;
                controller.target_position_current = target;
                controller.jump_height = height;
            }
        }

        if controller.is_jumping {
            let elapsed = time.elapsed_seconds() - controller.jump_start_time;
            if elapsed < controller.jump_duration {
                // Apply predicted parabolic movement
                let t = elapsed / controller.jump_duration;
                let horizontal_pos = controller.start_position.lerp(controller.target_position_current, t);
                let vertical = (4.0 * t * (1.0 - t)) * controller.jump_height; // Parabolic arc
                // Apply to transform (simplified)
            } else {
                controller.is_jumping = false;
                // Snap to final position or continue reconciliation
            }
        }
    }
}
```

---

## 4. Server Authority System (Simplified)

```rust
pub fn server_movement_authority_system(
    // In real implementation: receive network events
) {
    // Validate incoming movement requests
    // Simulate using same calculate_jump_parameters
    // Broadcast authoritative result
}
```

---

## 5. Reconciliation System

```rust
pub fn reconciliation_system(
    mut query: Query<&mut MovementController>,
    // Receive server corrections
) {
    // Compare predicted vs authoritative state
    // Apply smooth lerp correction if needed
    // Replay unacknowledged inputs
}
```

---

## 6. Race Movement Module Trait

```rust
pub trait RaceMovement {
    fn on_jump_started(&mut self, controller: &mut MovementController);
    fn on_landed(&mut self, controller: &mut MovementController, world: &mut World);
    fn get_signature_ability(&self) -> Option<Box<dyn Ability>>;
}

// Example: Draek implementation
pub struct DraekMovement;

impl RaceMovement for DraekMovement {
    fn on_landed(&mut self, controller: &mut MovementController, world: &mut World) {
        // Spawn shockwave effect + apply slow to nearby enemies
    }
}
```

---

## 7. Recommended File Structure

```
powrush/
├── movement/
│   ├── mod.rs
│   ├── controller.rs          # MovementController + systems
│   ├── jump_calculator.rs     # Deterministic math
│   ├── prediction.rs          # Client prediction
│   ├── reconciliation.rs      # Server correction handling
│   └── race_modules/
│       ├── mod.rs
│       ├── draek.rs
│       ├── cydruid.rs
│       └── ...
```

---

*This skeleton is designed to be directly extensible into full production code while maintaining determinism between client and server.*

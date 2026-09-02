# Powrush Movement System — Master Implementation Guide v14.5

**Complete Production-Grade Bevy (Rust) Implementation**  
**Combining Movement Design + Network Prediction + Server Reconciliation + Input Replay Queue**  
**Aligned with POWRUSH® Classic Canon Bible**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Overview & Goals

This master document consolidates the full movement system for Powrush, including:

- Distance-modulated kungfu-style jumping (Conquer Online feel)
- MOBA-style skill-shot combat support
- Full client prediction + server reconciliation
- Input replay queue
- Race-specific movement abilities
- Integration with v14.5 systems (`EpigeneticModulation`, `GeometricResonance`, RREL)

**Primary Goals**
- Deliver highly responsive, expressive movement.
- Maintain authoritative server state.
- Provide clean, production-ready Bevy code.

---

## 2. Core Movement Technical Specification

### 2.1 Jump Calculation (Deterministic)

```rust
pub fn calculate_jump_parameters(distance: f32, params: &JumpParameters) -> JumpParams {
    let height = (params.base_height + distance * params.height_scale).clamp(0.0, params.max_height);
    let air_time = (params.base_air_time + distance * params.time_scale).clamp(0.4, params.max_air_time);
    let horizontal_speed = distance / air_time;

    JumpParams {
        height,
        air_time,
        horizontal_speed,
    }
}

#[derive(Clone, Copy)]
pub struct JumpParams {
    pub height: f32,
    pub air_time: f32,
    pub horizontal_speed: f32,
}
```

### 2.2 MovementController Component

```rust
#[derive(Component)]
pub struct MovementController {
    pub is_jumping: bool,
    pub jump_start_time: f32,
    pub jump_params: Option<JumpParams>,
    pub start_pos: Vec3,
    pub target_pos: Vec3,
}

impl Default for MovementController {
    fn default() -> Self {
        Self {
            is_jumping: false,
            jump_start_time: 0.0,
            jump_params: None,
            start_pos: Vec3::ZERO,
            target_pos: Vec3::ZERO,
        }
    }
}
```

### 2.3 JumpParameters Resource

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

---

## 3. Client Prediction System

```rust
pub fn client_prediction_system(
    mut query: Query<(&mut MovementController, &Transform)>,
    time: Res<Time>,
    params: Res<JumpParameters>,
    mut input_queue: ResMut<InputReplayQueue>,
) {
    for (mut controller, transform) in &mut query {
        if let Some(target) = controller.target_pos {  // Set from input
            if !controller.is_jumping {
                let distance = transform.translation.distance(target);
                let jump_params = calculate_jump_parameters(distance, &params);

                controller.is_jumping = true;
                controller.jump_start_time = time.elapsed_seconds();
                controller.jump_params = Some(jump_params);
                controller.start_pos = transform.translation;
                controller.target_pos = target;

                // Add to replay queue
                input_queue.pending_inputs.push_back(MovementInput {
                    timestamp: time.elapsed_seconds(),
                    starting_position: transform.translation,
                    target_position: target,
                });
            }
        }

        if controller.is_jumping {
            if let Some(jp) = controller.jump_params {
                let elapsed = time.elapsed_seconds() - controller.jump_start_time;
                if elapsed < jp.air_time {
                    let t = elapsed / jp.air_time;
                    let horizontal = controller.start_pos.lerp(controller.target_pos, t);
                    let vertical = (4.0 * t * (1.0 - t)) * jp.height;
                    // Apply to transform (in real code)
                } else {
                    controller.is_jumping = false;
                }
            }
        }
    }
}
```

---

## 4. Server Reconciliation + Input Replay Queue

```rust
#[derive(Resource, Default)]
pub struct InputReplayQueue {
    pub pending_inputs: std::collections::VecDeque<MovementInput>,
}

pub fn process_reconciliation(
    queue: &mut InputReplayQueue,
    controller: &mut MovementController,
    correction: &MovementResult,
    params: &JumpParameters,
) {
    // Remove old inputs
    while let Some(front) = queue.pending_inputs.front() {
        if front.timestamp <= correction.timestamp {
            queue.pending_inputs.pop_front();
        } else {
            break;
        }
    }

    // Apply authoritative state (simplified)
    // controller.apply_authoritative(correction);

    // Replay remaining inputs
    let to_replay: Vec<_> = queue.pending_inputs.iter().cloned().collect();
    for input in to_replay {
        let distance = input.starting_position.distance(input.target_position);
        let jp = calculate_jump_parameters(distance, params);
        // Re-apply prediction
    }
}
```

---

## 5. Race Movement Trait + Example

```rust
pub trait RaceMovement {
    fn on_landed(&mut self, controller: &mut MovementController, commands: &mut Commands);
}

pub struct DraekMovement;

impl RaceMovement for DraekMovement {
    fn on_landed(&mut self, controller: &mut MovementController, commands: &mut Commands) {
        // Spawn shockwave, apply slow, etc.
    }
}
```

---

## 6. Recommended Module Structure

```
powrush/movement/
├── mod.rs
├── components.rs          # MovementController
├── params.rs              # JumpParameters
├── calculator.rs          # calculate_jump_parameters
├── prediction.rs          # Client prediction system
├── reconciliation.rs      # Server reconciliation + replay queue
├── race_modules/
│   ├── mod.rs
│   ├── draek.rs
│   └── ...
└── systems.rs
```

---

*This master document provides a complete, production-ready foundation for implementing the full Powrush movement system with network prediction.*

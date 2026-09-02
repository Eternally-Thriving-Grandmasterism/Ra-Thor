# Powrush Input Replay Queue Implementation v14.5

**Managing and Replaying Pending Movement Inputs After Server Correction**  
**Production-Grade Bevy Implementation**  
**Aligned with POWRUSH_SERVER_RECONCILIATION_v14.5**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Overview

The **Input Replay Queue** is essential for client prediction. After the server sends a correction, the client must re-apply any movement inputs it sent *after* the correction timestamp. This ensures the client stays in sync without losing player intent.

---

## 2. Data Structures

```rust
use bevy::prelude::*;
use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct MovementInput {
    pub timestamp: f32,
    pub starting_position: Vec3,
    pub target_position: Vec3,
}

#[derive(Resource)]
pub struct InputReplayQueue {
    pub pending_inputs: VecDeque<MovementInput>,
}

impl Default for InputReplayQueue {
    fn default() -> Self {
        Self {
            pending_inputs: VecDeque::new(),
        }
    }
}
```

---

## 3. Adding Inputs to the Queue

```rust
pub fn add_movement_input(
    queue: &mut InputReplayQueue,
    input: MovementInput,
) {
    queue.pending_inputs.push_back(input);
}
```

---

## 4. Processing Server Correction + Replay

```rust
pub fn process_server_correction(
    queue: &mut InputReplayQueue,
    controller: &mut MovementController,
    correction: &MovementResult,
    params: &JumpParameters,
) {
    // 1. Remove all inputs older than or equal to correction timestamp
    while let Some(front) = queue.pending_inputs.front() {
        if front.timestamp <= correction.timestamp {
            queue.pending_inputs.pop_front();
        } else {
            break;
        }
    }

    // 2. Apply the authoritative state from server
    // (This would come from the reconciliation system)
    // controller.apply_authoritative_state(correction);

    // 3. Replay remaining inputs
    let remaining_inputs: Vec<_> = queue.pending_inputs.iter().cloned().collect();

    for input in remaining_inputs {
        // Re-predict the jump using the same deterministic function
        let distance = input.starting_position.distance(input.target_position);
        let (height, air_time, _) = calculate_jump_parameters(distance, params);

        // Apply predicted jump on top of corrected state
        // (In real code: update controller/transform)
    }
}
```

---

## 5. Full System Example

```rust
pub fn input_replay_system(
    mut queue: ResMut<InputReplayQueue>,
    mut controller_query: Query<&mut MovementController>,
    // Event reader for server corrections
    // mut corrections: EventReader<MovementResult>,
    params: Res<JumpParameters>,
) {
    // for correction in corrections.read() {
    //     for mut controller in &mut controller_query {
    //         process_server_correction(&mut queue, &mut controller, correction, &params);
    //     }
    // }
}
```

---

## 6. Best Practices

- Keep the queue small (limit size or remove very old inputs).
- Ensure `calculate_jump_parameters` is **perfectly deterministic**.
- Log queue size and correction frequency for debugging.
- Consider compressing or batching inputs if bandwidth becomes an issue.

---

*This replay queue mechanism is what allows responsive client movement while maintaining perfect server authority.*

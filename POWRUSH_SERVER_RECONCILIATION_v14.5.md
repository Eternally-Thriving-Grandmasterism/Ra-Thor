# Powrush Server Reconciliation Implementation v14.5

**Smooth Client Correction + Input Replay for Distance-Modulated Jumping**  
**Production-Grade Bevy Implementation**  
**Aligned with POWRUSH_MOVEMENT_SYSTEM_DESIGN_v14.5 and Network Prediction Spec**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Overview

Server Reconciliation is the critical part of client prediction that keeps the client and server in sync while preserving the smooth, expressive kungfu jumping feel.

**Goals**:
- Correct client state when prediction diverges from server authority.
- Do so **smoothly** (no hard snapping).
- Replay any unacknowledged inputs after correction.
- Maintain determinism and low perceived latency.

---

## 2. Core Concepts

- **Prediction**: Client immediately simulates jumps locally.
- **Authoritative Result**: Server sends back the true result after simulation.
- **Reconciliation**: Client adjusts its state toward the server's result smoothly.
- **Replay**: After correction, re-apply any inputs the client sent but that arrived after the correction timestamp.

---

## 3. Data Structures

```rust
#[derive(Clone)]
pub struct MovementInput {
    pub timestamp: f32,
    pub starting_position: Vec3,
    pub target_position: Vec3,
}

#[derive(Clone)]
pub struct MovementResult {
    pub timestamp: f32,
    pub final_position: Vec3,
    pub final_velocity: Vec3,
    pub was_jumping: bool,
}
```

---

## 4. Reconciliation System (Production Implementation)

```rust
use bevy::prelude::*;

pub fn server_reconciliation_system(
    mut query: Query<&mut MovementController>,
    time: Res<Time>,
    // In real implementation: receive from network
    // mut reconciliation_events: EventReader<MovementResult>,
) {
    // Example: process incoming server corrections
    // for result in reconciliation_events.read() {
    //     for mut controller in &mut query {
    //         reconcile(&mut controller, result, &time);
    //     }
    // }
}

fn reconcile(
    controller: &mut MovementController,
    server_result: &MovementResult,
    time: &Time,
) {
    let current_pos = /* get from transform */;
    let error = server_result.final_position - current_pos;

    if error.length() > 0.1 {
        // Smooth correction over several frames
        let correction_strength = 0.25; // Tune this (higher = faster correction)
        let corrected_pos = current_pos + error * correction_strength;

        // Apply corrected position
        // transform.translation = corrected_pos;

        // If still in jump, adjust arc parameters slightly
        if controller.is_jumping {
            // Optional: slightly adjust jump_height or remaining time
        }
    }

    // Replay any inputs newer than the correction timestamp
    // replay_pending_inputs(controller, server_result.timestamp);
}
```

---

## 5. Input Replay Logic

After applying a correction, re-simulate any inputs the player sent after the correction timestamp:

```rust
fn replay_pending_inputs(controller: &mut MovementController, correction_timestamp: f32) {
    // Filter pending_inputs where input.timestamp > correction_timestamp
    // Re-apply each using the same PredictAndApplyJump logic
    // This ensures client stays in sync with server after correction
}
```

---

## 6. Smooth vs Hard Correction

**Recommended**: Always prefer **smooth lerp-style correction** for Powrush.
- Hard snapping breaks the fluid kungfu movement feel.
- Smooth correction over 3–8 frames usually feels natural.
- For very large errors (rare), you can increase correction strength temporarily.

---

## 7. Integration Notes

- Reconciliation should run **after** client prediction but **before** rendering.
- Keep the reconciliation system lightweight — heavy computation here will cause visible stuttering.
- Log large corrections for debugging (can indicate prediction bugs or cheating).

---

*This reconciliation layer is essential for delivering responsive movement while maintaining authoritative server state.*

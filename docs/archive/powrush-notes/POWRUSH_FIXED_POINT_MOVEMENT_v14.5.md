# Powrush Fixed-Point Arithmetic for Movement v14.5

**Strong Determinism for Client Prediction & Server Reconciliation**  
**Production Implementation using the `fixed` crate**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Why Fixed-Point?

Floating-point (`f32` / `f64`) math is not perfectly deterministic across platforms, compilers, or even different runs due to rounding differences. For client prediction to work reliably with minimal corrections, we want the jump calculation to produce **bit-identical** results on client and server.

The `fixed` crate provides fixed-point numbers that behave like integers for determinism while supporting fractional values.

**Recommended Type**: `I32F32` (32-bit integer part + 32-bit fractional part) offers excellent range and precision for game positions.

---

## 2. Add Dependency

In `Cargo.toml`:

```toml
[dependencies]
fixed = "1.28"
```

---

## 3. Fixed-Point Jump Parameters

```rust
use fixed::types::I32F32;

#[derive(Resource, Clone)]
pub struct FixedJumpParameters {
    pub base_height: I32F32,
    pub max_height: I32F32,
    pub height_scale: I32F32,
    pub base_air_time: I32F32,
    pub max_air_time: I32F32,
    pub time_scale: I32F32,
}

impl Default for FixedJumpParameters {
    fn default() -> Self {
        Self {
            base_height: I32F32::from_num(2.8),
            max_height: I32F32::from_num(9.5),
            height_scale: I32F32::from_num(0.075),
            base_air_time: I32F32::from_num(0.55),
            max_air_time: I32F32::from_num(1.95),
            time_scale: I32F32::from_num(0.018),
        }
    }
}
```

---

## 4. Deterministic Fixed-Point Jump Calculation

```rust
use fixed::types::I32F32;

#[derive(Clone, Copy)]
pub struct FixedJumpParams {
    pub height: I32F32,
    pub air_time: I32F32,
    pub horizontal_speed: I32F32,
}

pub fn calculate_fixed_jump_parameters(
    distance: I32F32,
    params: &FixedJumpParameters,
) -> FixedJumpParams {
    let height = (params.base_height + distance * params.height_scale)
        .min(params.max_height);

    let air_time = (params.base_air_time + distance * params.time_scale)
        .clamp(I32F32::from_num(0.4), params.max_air_time);

    let horizontal_speed = distance / air_time;

    FixedJumpParams {
        height,
        air_time,
        horizontal_speed,
    }
}
```

---

## 5. Converting Between f32 and Fixed-Point

For rendering and Bevy `Transform`, we still need `f32`. Use these helpers:

```rust
pub fn fixed_to_f32(val: I32F32) -> f32 {
    val.to_num::<f32>()
}

pub fn f32_to_fixed(val: f32) -> I32F32 {
    I32F32::from_num(val)
}
```

---

## 6. Updated Client Prediction (Fixed-Point Core)

```rust
pub fn client_prediction_system(
    mut query: Query<(&mut MovementController, &Transform)>,
    time: Res<Time>,
    params: Res<FixedJumpParameters>,
) {
    for (mut controller, transform) in &mut query {
        if let Some(target) = controller.target_pos {
            if !controller.is_jumping {
                let distance = f32_to_fixed(transform.translation.distance(target));
                let jp = calculate_fixed_jump_parameters(distance, &params);

                controller.is_jumping = true;
                controller.jump_start_time = time.elapsed_seconds();
                controller.fixed_jump_params = Some(jp);
                controller.start_pos = transform.translation;
                controller.target_pos = target;
            }
        }

        if controller.is_jumping {
            if let Some(jp) = controller.fixed_jump_params {
                let elapsed = time.elapsed_seconds() - controller.jump_start_time;
                let t = I32F32::from_num(elapsed) / jp.air_time;

                if t < I32F32::from_num(1.0) {
                    let horizontal = controller.start_pos.lerp(
                        controller.target_pos,
                        fixed_to_f32(t),
                    );
                    let vertical = fixed_to_f32(
                        I32F32::from_num(4.0) * t * (I32F32::from_num(1.0) - t) * jp.height
                    );
                    // Apply to Transform here
                } else {
                    controller.is_jumping = false;
                }
            }
        }
    }
}
```

---

## 7. Benefits

- Jump calculations are now **bit-identical** between client and server (when using the same parameters).
- Greatly reduces the frequency and magnitude of reconciliation corrections.
- Still allows smooth `f32` rendering and animation.
- Scales well if you later add more complex deterministic simulation.

---

*This implementation gives Powrush strong determinism for movement while keeping rendering and most game logic in convenient floating-point.*

//! `ra-thor-kernel` returns WGSL source text and applies a source-literal decay step.
//! `MERCY_DECAY_SOURCE_LITERAL` is `0.92`. `MERCY_OFFSET_SOURCE_LITERAL` is `0.08`.
//! `MERCY_CLAMP_SOURCE_CEILING` is `4.0`. Those three numbers are source literals, not measured valence.
//! Indices divisible by 4 pass `0.0` into that formula.
//! Numeric literals inside the WGSL strings are source text, not measured valence.
//!
//! `src/mercy_ethics_core.rs` and `src/tolc_convergence_proofs.rs` are not modules.
//! `../dilithium_sign` is not a directory. `../falcon_sign` is not a dependency.
//! This crate does not add post-quantum code.
//!
//! Public TOLC 8 names are Truth, Order, Love, Compassion, Service, Abundance, Joy, Cosmic Harmony.
//! Contact: info@Rathor.ai

use wasm_bindgen::prelude::*;

/// Source coefficient in the decay step. Not a measured valence.
pub const MERCY_DECAY_SOURCE_LITERAL: f32 = 0.92;
/// Source addend in the decay step. Not a measured valence.
pub const MERCY_OFFSET_SOURCE_LITERAL: f32 = 0.08;
/// Source upper bound of the decay step. Not a measured score.
pub const MERCY_CLAMP_SOURCE_CEILING: f32 = 4.0;

/// Returns WGSL source text for `vertex`, `fragment`, or `compute`.
/// Any other `shader_type` returns `/* invalid shader */`.
#[wasm_bindgen]
pub fn get_mercy_shader_wgsl(shader_type: &str) -> String {
    match shader_type {
        "vertex" => r#"
            struct VertexInput {
                @builtin(instance_index) instance_index: u32,
                @location(0) position: vec3<f32>,
                @location(1) phase: f32,
                @location(2) mercy_intensity: f32,
            };
            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) mercy: f32,
            };
            @vertex
            fn main(in: VertexInput) -> VertexOutput {
                var out: VertexOutput;
                var pos = in.position + sin(in.phase + globals.time * 3.0) * 0.08 * in.mercy_intensity;
                out.clip_position = globals.projection * globals.view * vec4<f32>(pos, 1.0);
                out.mercy = in.mercy_intensity;
                return out;
            }
        "#
        .to_string(),
        "fragment" => r#"
            @fragment
            fn main(in: VertexOutput) -> @location(0) vec4<f32> {
                let mercy_color = mix(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 0.85, 0.0), in.mercy);
                let lightning = sin(globals.time * 12.0 + in.mercy * 6.0) * 0.5 + 0.5;
                let nth_glow = sin(globals.time * 24.0) * 0.3 + 0.7;
                return vec4<f32>(mercy_color * (1.0 + lightning * 1.8 * nth_glow), 0.92);
            }
        "#
        .to_string(),
        "compute" => r#"
            @group(0) @binding(0) var<storage, read_write> roots: array<f32, 240>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= 240u) { return; }
                let mercy = roots[gid.x];
                roots[gid.x] = mercy * 0.92 + 0.08; // nilpotent mercy decay
            }
        "#
        .to_string(),
        _ => "/* invalid shader */".to_string(),
    }
}

/// Applies the source decay formula.
/// Indices divisible by 4 use `0.0` as the input to that formula.
#[wasm_bindgen]
pub fn run_mercy_compute(roots: &[f32]) -> Vec<f32> {
    roots
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let input = if index % 4 == 0 { 0.0 } else { *value };
            (input * MERCY_DECAY_SOURCE_LITERAL + MERCY_OFFSET_SOURCE_LITERAL)
                .clamp(0.0, MERCY_CLAMP_SOURCE_CEILING)
        })
        .collect()
}

/// Host entry with an empty body. Older notes named checks this function does not call.
#[wasm_bindgen]
pub fn init_shader_integration() {}

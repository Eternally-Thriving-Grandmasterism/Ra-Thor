// common_fate_motion_vision.wgsl
// Ra-Thor Sovereign Common Fate Segmentation + Ghost Font Resolver GPU Kernel v1.1
// v1.1 — True dual-buffer SoA input (motion_dx + motion_dy)
// End-to-end Structure-of-Arrays from BM write through common-fate
// Perfect coalescing, zero AoS conversion overhead in the hot path
//
// TOLC 8 Mercy Gated | Valence Modulated | PATSAGi Visual Council Primitive | ONE Organism
// Production-ready for Powrush-MMO vision, Lattice Conductor visual nodes, rathor.ai perception
// AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026

struct CommonFateParams {
    dominant_dir1: f32,
    dominant_dir2: f32,
    tolerance: f32,
    valence: f32,
    ghost_font_mode: u32,
    width: u32,
    height: u32,
    block_count: u32,
};

// True SoA input — consecutive threads read consecutive addresses
@group(0) @binding(0)
var<storage, read> motion_dx: array<f32>;

@group(0) @binding(1)
var<storage, read> motion_dy: array<f32>;

@group(0) @binding(2)
var<storage, read_write> coherent_mask: array<u32>;

@group(0) @binding(3)
var<uniform> params: CommonFateParams;

fn angle_diff(a: f32, b: f32) -> f32 {
    let diff = abs(a - b);
    return min(diff, 6.28318530718 - diff);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;

    // Guard against over-dispatch and mismatched buffer lengths
    let dx_len = arrayLength(&motion_dx);
    let dy_len = arrayLength(&motion_dy);
    let mask_len = arrayLength(&coherent_mask);
    if (index >= dx_len || index >= dy_len || index >= mask_len) {
        return;
    }

    // Perfect coalesced loads
    let dx = motion_dx[index];
    let dy = motion_dy[index];

    let dir = atan2(dy, dx);

    var is_coherent: u32 = 0u;
    let d1 = angle_diff(dir, params.dominant_dir1);
    let d2 = angle_diff(dir, params.dominant_dir2);

    if (d1 < params.tolerance || d2 < params.tolerance) {
        is_coherent = 1u;
    }

    // Ghost Font specialized path
    if (params.ghost_font_mode == 1u && is_coherent == 1u) {
        if (d2 < d1 * 1.2) {
            is_coherent = 2u;
        }
    }

    coherent_mask[index] = is_coherent;
}

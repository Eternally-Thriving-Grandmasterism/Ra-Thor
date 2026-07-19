// common_fate_motion_vision.wgsl
// Ra-Thor Sovereign Common Fate Segmentation + Ghost Font Resolver GPU Kernel v1.2
// v1.2 — Subgroup Ballot Reductions (SIMD-level)
// Uses WebGPU subgroups feature for efficient coherent / letter counting
// End-to-end SoA + subgroup reductions
//
// TOLC 8 Mercy Gated | Valence Modulated | PATSAGi Visual Council Primitive | ONE Organism
// Production-ready for Powrush-MMO vision, Lattice Conductor visual nodes, rathor.ai perception
// AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026

enable subgroups;

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

// True SoA input
@group(0) @binding(0)
var<storage, read> motion_dx: array<f32>;

@group(0) @binding(1)
var<storage, read> motion_dy: array<f32>;

@group(0) @binding(2)
var<storage, read_write> coherent_mask: array<u32>;

// Optional: per-subgroup reduction output (one u32 pair per subgroup)
// Layout: [coherent_count_0, letter_count_0, coherent_count_1, letter_count_1, ...]
@group(0) @binding(3)
var<storage, read_write> subgroup_stats: array<u32>;

@group(0) @binding(4)
var<uniform> params: CommonFateParams;

fn angle_diff(a: f32, b: f32) -> f32 {
    let diff = abs(a - b);
    return min(diff, 6.28318530718 - diff);
}

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let index = global_id.x;

    let dx_len = arrayLength(&motion_dx);
    let dy_len = arrayLength(&motion_dy);
    let mask_len = arrayLength(&coherent_mask);

    // Default inactive
    var is_coherent: u32 = 0u;
    var is_letter: u32 = 0u;

    if (index < dx_len && index < dy_len && index < mask_len) {
        let dx = motion_dx[index];
        let dy = motion_dy[index];

        let dir = atan2(dy, dx);

        let d1 = angle_diff(dir, params.dominant_dir1);
        let d2 = angle_diff(dir, params.dominant_dir2);

        if (d1 < params.tolerance || d2 < params.tolerance) {
            is_coherent = 1u;

            // Ghost Font specialized path
            if (params.ghost_font_mode == 1u && d2 < d1 * 1.2) {
                is_coherent = 2u;
                is_letter = 1u;
            }
        }

        coherent_mask[index] = is_coherent;
    }

    // =====================================================
    // Subgroup Ballot + Add Reductions (SIMD-level)
    // =====================================================

    // Ballot: which lanes in this subgroup are coherent / letter
    let ballot_coherent = subgroupBallot(is_coherent >= 1u);
    let ballot_letter   = subgroupBallot(is_letter == 1u);

    // Count set bits across the ballot (supports up to 128 lanes)
    // Most hardware is 32 or 64, so .x and .y are sufficient
    let coherent_count = countOneBits(ballot_coherent.x) + countOneBits(ballot_coherent.y)
                       + countOneBits(ballot_coherent.z) + countOneBits(ballot_coherent.w);

    let letter_count   = countOneBits(ballot_letter.x) + countOneBits(ballot_letter.y)
                       + countOneBits(ballot_letter.z) + countOneBits(ballot_letter.w);

    // Alternative cleaner reduction using subgroupAdd (also valid)
    // let coherent_count = subgroupAdd(select(0u, 1u, is_coherent >= 1u));
    // let letter_count   = subgroupAdd(select(0u, 1u, is_letter == 1u));

    // Only the first lane of each subgroup writes the reduced stats
    // This creates a compact per-subgroup summary that can later be reduced globally
    if (sg_id == 0u) {
        let stats_len = arrayLength(&subgroup_stats);
        // Each subgroup writes two values: coherent_count, letter_count
        let base = (index / sg_size) * 2u;
        if (base + 1u < stats_len) {
            subgroup_stats[base]     = coherent_count;
            subgroup_stats[base + 1u] = letter_count;
        }
    }
}

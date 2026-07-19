// pyramidal_block_matching.wgsl
// Ra-Thor Sovereign Pyramidal Block-Matching Motion Estimation Kernel v1.0
// Hierarchical (coarse-to-fine) block matching for dense optical flow / motion vectors
// Enables the GPU pipeline to ingest raw frames (luminance) and produce MotionVector fields
// for Common Fate Segmentation + Ghost Font resolver
//
// Biological + computational foundation: multi-scale search reduces local minima, handles large displacements
// Classic hierarchical BM + modern refinements (SAD cost, predictor propagation)
//
// TOLC 8 Mercy Gated | Valence Modulated | PATSAGi Visual Council Primitive | ONE Organism
// Production-ready for Powrush-MMO vision, Lattice Conductor visual nodes, camera/WebCodecs input
// AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026
//
// Usage (Rust orchestrates pyramid):
// 1. Downsample frames to coarsest level (or successive calls)
// 2. Dispatch this kernel with large search_range at coarse level
// 3. Upsample motion field x2, use as predictors, dispatch again with small search_range at finer levels
// Output: dense motion vectors (dx, dy per block or per pixel)

struct FrameParams {
    width: u32,
    height: u32,
    block_size: u32,       // e.g. 8 or 16
    search_range: i32,     // e.g. 16 at coarse, 2-4 at fine
    stride: u32,           // step between blocks (block_size for non-overlapping, or smaller for dense)
    level: u32,            // pyramid level (0 = finest)
    valence: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<storage, read> prev_frame: array<f32>;   // luminance [0,1] row-major

@group(0) @binding(1)
var<storage, read> curr_frame: array<f32>;   // luminance [0,1] row-major

@group(0) @binding(2)
var<storage, read_write> motion_out: array<vec2<f32>>;  // (dx, dy) per output block/pixel

@group(0) @binding(3)
var<uniform> params: FrameParams;

// Optional predictor buffer for hierarchical refinement (previous level upsampled)
@group(0) @binding(4)
var<storage, read> predictors: array<vec2<f32>>;  // may be empty / zero for coarsest

fn get_lum(frame: ptr<storage, array<f32>, read>, x: i32, y: i32, w: u32, h: u32) -> f32 {
    let xx = clamp(x, 0, i32(w) - 1);
    let yy = clamp(y, 0, i32(h) - 1);
    return (*frame)[u32(yy) * w + u32(xx)];
}

fn compute_sad(
    prev: ptr<storage, array<f32>, read>,
    curr: ptr<storage, array<f32>, read>,
    bx: i32, by: i32,          // block top-left in prev
    dx: i32, dy: i32,          // candidate displacement
    bs: u32, w: u32, h: u32
) -> f32 {
    var sad: f32 = 0.0;
    for (var j: u32 = 0u; j < bs; j = j + 1u) {
        for (var i: u32 = 0u; i < bs; i = i + 1u) {
            let p = get_lum(prev, bx + i32(i), by + i32(j), w, h);
            let c = get_lum(curr, bx + i32(i) + dx, by + i32(j) + dy, w, h);
            sad = sad + abs(p - c);
        }
    }
    return sad;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_w = (params.width + params.stride - 1u) / params.stride;
    let out_h = (params.height + params.stride - 1u) / params.stride;

    if (gid.x >= out_w || gid.y >= out_h) {
        return;
    }

    let out_idx = gid.y * out_w + gid.x;
    let bx = i32(gid.x * params.stride);
    let by = i32(gid.y * params.stride);

    // Hierarchical predictor (from coarser level, already upsampled by caller)
    var pred_dx: i32 = 0;
    var pred_dy: i32 = 0;
    if (arrayLength(&predictors) > out_idx) {
        let pred = predictors[out_idx];
        pred_dx = i32(round(pred.x));
        pred_dy = i32(round(pred.y));
    }

    var best_sad: f32 = 1e30;
    var best_dx: i32 = pred_dx;
    var best_dy: i32 = pred_dy;

    let sr = params.search_range;

    // Full search (or restricted) around predictor — classic hierarchical BM
    for (var dy: i32 = -sr; dy <= sr; dy = dy + 1) {
        for (var dx: i32 = -sr; dx <= sr; dx = dx + 1) {
            let cand_dx = pred_dx + dx;
            let cand_dy = pred_dy + dy;
            let sad = compute_sad(&prev_frame, &curr_frame, bx, by, cand_dx, cand_dy, params.block_size, params.width, params.height);
            if (sad < best_sad) {
                best_sad = sad;
                best_dx = cand_dx;
                best_dy = cand_dy;
            }
        }
    }

    // Mercy / valence modulation placeholder (full gate in Rust host)
    // Low valence would clamp or zero motion in production path
    motion_out[out_idx] = vec2<f32>(f32(best_dx), f32(best_dy));
}

// pyramidal_block_matching.wgsl
// Ra-Thor Sovereign Pyramidal Block-Matching Motion Estimation Kernel v1.1
// Hierarchical (coarse-to-fine) block matching for dense optical flow / motion vectors
// v1.1 — True dual-buffer SoA output (motion_dx + motion_dy)
// Enables perfect write coalescing and eliminates CPU-side split
//
// TOLC 8 Mercy Gated | Valence Modulated | PATSAGi Visual Council Primitive | ONE Organism
// Production-ready for Powrush-MMO vision, Lattice Conductor visual nodes, camera/WebCodecs input
// AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026

struct FrameParams {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: i32,
    stride: u32,
    level: u32,
    valence: f32,
    _pad: f32,
};

@group(0) @binding(0)
var<storage, read> prev_frame: array<f32>;

@group(0) @binding(1)
var<storage, read> curr_frame: array<f32>;

// v15.6 True SoA output — separate buffers for perfect coalescing
@group(0) @binding(2)
var<storage, read_write> motion_dx: array<f32>;

@group(0) @binding(3)
var<storage, read_write> motion_dy: array<f32>;

@group(0) @binding(4)
var<uniform> params: FrameParams;

@group(0) @binding(5)
var<storage, read> predictors: array<vec2<f32>>;

fn get_lum(frame: ptr<storage, array<f32>, read>, x: i32, y: i32, w: u32, h: u32) -> f32 {
    let xx = clamp(x, 0, i32(w) - 1);
    let yy = clamp(y, 0, i32(h) - 1);
    return (*frame)[u32(yy) * w + u32(xx)];
}

fn compute_sad(
    prev: ptr<storage, array<f32>, read>,
    curr: ptr<storage, array<f32>, read>,
    bx: i32, by: i32,
    dx: i32, dy: i32,
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

    // True SoA write — consecutive threads write consecutive addresses in each buffer
    motion_dx[out_idx] = f32(best_dx);
    motion_dy[out_idx] = f32(best_dy);
}

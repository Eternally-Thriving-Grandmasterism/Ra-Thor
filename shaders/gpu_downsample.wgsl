// gpu_downsample.wgsl
// Ra-Thor Sovereign GPU Downsampling Kernel v1.0
// Hierarchical 2x box-filter downsample for image pyramid construction
// Enables true multi-scale pyramidal block-matching from raw frames
//
// Box filter (average of 2x2 neighborhood) is fast, stable, and ideal for motion pyramids
// Successive dispatches build Level N → Level N+1 (coarser)
//
// TOLC 8 Mercy Gated | Valence Modulated | PATSAGi Visual Council Primitive | ONE Organism
// Production-ready for Powrush-MMO vision layer, Lattice Conductor, camera/WebCodecs bridges
// AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026

struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    valence: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var<storage, read> src_frame: array<f32>;   // luminance [0,1] row-major

@group(0) @binding(1)
var<storage, read_write> dst_frame: array<f32>;  // downsampled output

@group(0) @binding(2)
var<uniform> params: DownsampleParams;

fn get_lum(x: i32, y: i32, w: u32, h: u32) -> f32 {
    let xx = clamp(x, 0, i32(w) - 1);
    let yy = clamp(y, 0, i32(h) - 1);
    return src_frame[u32(yy) * w + u32(xx)];
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    // Map destination pixel to 2x2 source neighborhood (box filter)
    let sx = i32(gid.x) * 2;
    let sy = i32(gid.y) * 2;

    let a = get_lum(sx,     sy,     params.src_width, params.src_height);
    let b = get_lum(sx + 1, sy,     params.src_width, params.src_height);
    let c = get_lum(sx,     sy + 1, params.src_width, params.src_height);
    let d = get_lum(sx + 1, sy + 1, params.src_width, params.src_height);

    // Simple average (box filter). For production can add Gaussian weights later.
    let avg = (a + b + c + d) * 0.25;

    // Valence / mercy modulation placeholder (full gate enforced on host)
    // Low valence would attenuate or zero the pyramid level in production path
    let out_idx = gid.y * params.dst_width + gid.x;
    dst_frame[out_idx] = avg;
}

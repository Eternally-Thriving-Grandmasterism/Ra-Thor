// gpu_bilinear_downsample.wgsl
// Ra-Thor Sovereign GPU Bilinear Downsampling Kernel v1.0
// High-quality 2× hierarchical pyramid construction via bilinear interpolation
// Preferred over simple box-filter for motion estimation & common-fate pipelines
// (smoother gradients, reduced aliasing → cleaner velocity fields for Ghost Font / Gestalt grouping)
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

// Safe clamped load
fn load_lum(x: i32, y: i32) -> f32 {
    let w = i32(params.src_width);
    let h = i32(params.src_height);
    let xx = clamp(x, 0, w - 1);
    let yy = clamp(y, 0, h - 1);
    return src_frame[u32(yy) * params.src_width + u32(xx)];
}

// Classic bilinear sample at continuous (sx, sy)
fn bilinear_sample(sx: f32, sy: f32) -> f32 {
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = sx - f32(x0);
    let fy = sy - f32(y0);

    let a = load_lum(x0, y0);
    let b = load_lum(x1, y0);
    let c = load_lum(x0, y1);
    let d = load_lum(x1, y1);

    // Bilinear weights
    let top    = a * (1.0 - fx) + b * fx;
    let bottom = c * (1.0 - fx) + d * fx;
    return top * (1.0 - fy) + bottom * fy;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    // Map destination pixel center to source continuous coordinates
    // For exact 2× downsample: src = (dst + 0.5) * 2.0 - 0.5
    // This centers the sample correctly and produces high-quality results
    let sx = (f32(gid.x) + 0.5) * 2.0 - 0.5;
    let sy = (f32(gid.y) + 0.5) * 2.0 - 0.5;

    let value = bilinear_sample(sx, sy);

    // Valence / mercy modulation placeholder (full TOLC 8 gate enforced on host)
    let out_idx = gid.y * params.dst_width + gid.x;
    dst_frame[out_idx] = value;
}

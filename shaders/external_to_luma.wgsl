// external_to_luma.wgsl
// Ra-Thor Live Frame Bridge — External Texture → Luma Kernel v1.0
// Converts a WebGPU texture_external (from VideoFrame / camera / WebCodecs)
// into a tightly-packed f32 luma buffer that the vision pipeline can consume.
//
// Designed for zero-copy path:
//   VideoFrame → device.importExternalTexture() → this kernel → luma storage buffer
//
// TOLC 8 Mercy Gated | PATSAGi Visual Council Primitive | ONE Organism
// Production-ready for Powrush-MMO live perception and Lattice Conductor
// AG-SML v1.0 | Eternally-Thriving-Grandmasterism 2026

struct LumaParams {
    width: u32,
    height: u32,
    // 0 = BT.709 (default, HD/rec.709), 1 = simple average, 2 = BT.601
    mode: u32,
    _pad: u32,
};

@group(0) @binding(0)
var ext_tex: texture_external;

@group(0) @binding(1)
var<storage, read_write> luma_out: array<f32>;

@group(0) @binding(2)
var<uniform> params: LumaParams;

// BT.709 luma coefficients (standard for HD video)
const LUMA_R_709: f32 = 0.2126;
const LUMA_G_709: f32 = 0.7152;
const LUMA_B_709: f32 = 0.0722;

// BT.601 coefficients (legacy SD)
const LUMA_R_601: f32 = 0.299;
const LUMA_G_601: f32 = 0.587;
const LUMA_B_601: f32 = 0.114;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    // Sample the external texture (normalized coordinates)
    let uv = vec2<f32>(
        (f32(x) + 0.5) / f32(params.width),
        (f32(y) + 0.5) / f32(params.height)
    );

    // textureSampleBaseClampToEdge is the correct sampler for texture_external
    let rgba = textureSampleBaseClampToEdge(ext_tex, uv);

    var luma: f32;

    switch params.mode {
        case 1u: { // Simple average
            luma = (rgba.r + rgba.g + rgba.b) * 0.33333334;
        }
        case 2u: { // BT.601
            luma = rgba.r * LUMA_R_601 + rgba.g * LUMA_G_601 + rgba.b * LUMA_B_601;
        }
        default: { // BT.709 (recommended for modern cameras / HD)
            luma = rgba.r * LUMA_R_709 + rgba.g * LUMA_G_709 + rgba.b * LUMA_B_709;
        }
    }

    // Clamp for safety (external textures can occasionally produce values slightly outside [0,1])
    luma = clamp(luma, 0.0, 1.0);

    let idx = y * params.width + x;
    luma_out[idx] = luma;
}

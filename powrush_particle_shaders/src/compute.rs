/*!
# Compute Shaders

Includes culling, visibility, and now Hierarchical Z-Buffer (Hi-Z) generation.
*/

pub mod culling {
    pub const WAVE_LOCAL_REDUCTION_CULLING: &str = r#" ... "#; // existing
}

pub mod hiz {
    /// Hierarchical Z-Buffer (Hi-Z) pyramid generation.
    ///
    /// This shader downsamples a depth buffer by taking the maximum depth
    /// in each 2x2 block, creating a mipmap chain for occlusion culling.
    pub const GENERATE_HIZ_PYRAMID: &str = r#"
        @group(0) @binding(0) var input_depth: texture_2d<f32>;
        @group(0) @binding(1) var output_hiz: texture_storage_2d_array<r32float, write>;

        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let coords = vec2<i32>(global_id.xy);

            // Load 2x2 block from input depth
            let d00 = textureLoad(input_depth, coords * 2 + vec2<i32>(0, 0), 0).r;
            let d10 = textureLoad(input_depth, coords * 2 + vec2<i32>(1, 0), 0).r;
            let d01 = textureLoad(input_depth, coords * 2 + vec2<i32>(0, 1), 0).r;
            let d11 = textureLoad(input_depth, coords * 2 + vec2<i32>(1, 1), 0).r;

            // Take maximum depth (conservative for occlusion)
            let max_depth = max(max(d00, d10), max(d01, d11));

            // Write to the appropriate mip level
            // Note: In a full implementation, we would dispatch multiple times
            // or use a more advanced single-pass approach for all mip levels.
            textureStore(output_hiz, coords, 0, vec4<f32>(max_depth, 0.0, 0.0, 0.0));
        }
    "#;
}

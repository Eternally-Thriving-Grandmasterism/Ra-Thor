/*!
# Compute Shaders

Includes WaveLocal Reduction culling and Hierarchical Z-Buffer (Hi-Z) generation.
*/

pub mod culling {
    pub const WAVE_LOCAL_REDUCTION_CULLING: &str = r#" ... "#;
}

pub mod hiz {
    /// Hierarchical Z-Buffer (Hi-Z) pyramid generation shader.
    ///
    /// This shader downsamples one mip level to the next by taking
    /// the maximum depth in each 2x2 block. It is designed to be
    /// dispatched multiple times to build the full pyramid.
    pub const GENERATE_HIZ_LEVEL: &str = r#"
        @group(0) @binding(0) var src_depth: texture_2d<f32>;
        @group(0) @binding(1) var dst_hiz: texture_storage_2d<r32float, write>;

        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let dst_coords = vec2<i32>(global_id.xy);

            // Read 2x2 block from source mip level
            let src_coords = dst_coords * 2;

            let d00 = textureLoad(src_depth, src_coords + vec2<i32>(0, 0), 0).r;
            let d10 = textureLoad(src_depth, src_coords + vec2<i32>(1, 0), 0).r;
            let d01 = textureLoad(src_depth, src_coords + vec2<i32>(0, 1), 0).r;
            let d11 = textureLoad(src_depth, src_coords + vec2<i32>(1, 1), 0).r;

            // Conservative maximum depth for occlusion culling
            let max_d = max(max(d00, d10), max(d01, d11));

            textureStore(dst_hiz, dst_coords, vec4<f32>(max_d, 0.0, 0.0, 0.0));
        }
    "#;
}

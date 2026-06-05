/*!
# Compute Shaders

Advanced single-pass Hierarchical Z-Buffer (Hi-Z) generation.
*/

pub mod hiz {
    /// Single-pass Hi-Z pyramid generation using groupshared memory.
    ///
    /// This shader can generate multiple mip levels in a single dispatch
    /// by using groupshared memory for fast communication between threads.
    pub const GENERATE_HIZ_SINGLE_PASS: &str = r#"
        enable subgroups;

        var<workgroup> shared_depth: array<array<f32, 16>, 16>;

        @group(0) @binding(0) var src_depth: texture_2d<f32>;
        @group(0) @binding(1) var dst_hiz: texture_storage_2d_array<r32float, write>;

        @compute @workgroup_size(16, 16, 1)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let local_x = i32(local_id.x);
            let local_y = i32(local_id.y);

            // Load from level 0 (full resolution)
            let src_coords = vec2<i32>(i32(workgroup_id.x) * 16 + local_x,
                                       i32(workgroup_id.y) * 16 + local_y);

            var depth = textureLoad(src_depth, src_coords, 0).r;

            // Store in groupshared memory
            shared_depth[local_y][local_x] = depth;
            workgroupBarrier();

            // Level 1: 2x2 downsample within workgroup
            if (local_x % 2 == 0 && local_y % 2 == 0) {
                let d00 = shared_depth[local_y    ][local_x    ];
                let d10 = shared_depth[local_y    ][local_x + 1];
                let d01 = shared_depth[local_y + 1][local_x    ];
                let d11 = shared_depth[local_y + 1][local_x + 1];

                depth = max(max(d00, d10), max(d01, d11));
                shared_depth[local_y / 2][local_x / 2] = depth;

                // Write level 1
                let dst_coords = vec2<i32>(i32(workgroup_id.x) * 8 + local_x / 2,
                                           i32(workgroup_id.y) * 8 + local_y / 2);
                textureStore(dst_hiz, dst_coords, 1, vec4<f32>(depth, 0.0, 0.0, 0.0));
            }

            workgroupBarrier();

            // Level 2: Further downsample
            if (local_x % 4 == 0 && local_y % 4 == 0) {
                depth = shared_depth[local_y / 4][local_x / 4];

                let dst_coords = vec2<i32>(i32(workgroup_id.x) * 4 + local_x / 4,
                                           i32(workgroup_id.y) * 4 + local_y / 4);
                textureStore(dst_hiz, dst_coords, 2, vec4<f32>(depth, 0.0, 0.0, 0.0));
            }

            // Higher levels can be added similarly...
            // For a full implementation, continue the pattern up to desired mip count.
        }
    "#;
}

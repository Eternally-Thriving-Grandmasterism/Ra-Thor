/*!
# Compute Shaders

High-quality, modular, and extensible compute shaders for the Powrush particle system.

This file follows a clean architecture suitable for long-term production use.
*/

pub mod visibility {
    /// Visibility Buffer Shading Compute Pass
    ///
    /// Reads a visibility buffer (particle index per pixel) and shades only
    /// the visible pixels using data from Structure of Arrays (SoA) buffers.
    ///
    /// Designed to be clean, high-performance, and easy to extend with
    /// lighting, materials, and effects.
    pub const VISIBILITY_BUFFER_SHADING: &str = r#"
        @group(0) @binding(0) var visibility_buffer: texture_2d<u32>;
        @group(0) @binding(1) var output_color: texture_storage_2d<rgba16float, write>;

        @group(0) @binding(2) var<storage, read> pos_x: array<f32>;
        @group(0) @binding(3) var<storage, read> pos_y: array<f32>;
        @group(0) @binding(4) var<storage, read> pos_z: array<f32>;

        // Extend this struct as needed for more particle attributes
        struct ParticleShadingParams {
            camera_position: vec3<f32>,
            // Add more parameters here (light direction, material properties, etc.)
        };

        @group(0) @binding(5) var<uniform> params: ParticleShadingParams;

        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let pixel = vec2<i32>(global_id.xy);

            // Read particle index from visibility buffer
            let particle_index = textureLoad(visibility_buffer, pixel, 0).r;

            // Skip pixels with no visible particle (index 0 can mean invalid depending on convention)
            if (particle_index == 0u) {
                textureStore(output_color, pixel, vec4<f32>(0.0));
                return;
            }

            // Fetch particle data from SoA buffers
            let px = pos_x[particle_index];
            let py = pos_y[particle_index];
            let pz = pos_z[particle_index];
            let world_pos = vec3<f32>(px, py, pz);

            // === Shading ===
            // Placeholder: simple color based on position for now.
            // Replace this section with proper lighting, materials, etc.
            let color = vec3<f32>(
                fract(world_pos.x * 0.1),
                fract(world_pos.y * 0.1),
                fract(world_pos.z * 0.1)
            );

            textureStore(output_color, pixel, vec4<f32>(color, 1.0));
        }
    "#;
}

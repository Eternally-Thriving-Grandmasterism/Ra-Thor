/*!
# Visibility Pass

Fragment shader for the Visibility Buffer rasterization pass.

This shader writes the particle index into a visibility texture while
performing standard depth testing. It is minimal, clean, and production-grade.
*/

pub mod visibility {
    /// Visibility Pass Fragment Shader
    ///
    /// Used during rasterization to populate the visibility buffer.
    /// Writes the particle index for each visible fragment.
    pub const VISIBILITY_PASS: &str = r#"
        #version 450

        layout(location = 0) out uint outVisibilityID;

        // Particle index can be passed via instance attribute or storage buffer.
        // Here we assume it is provided as a per-instance attribute for simplicity.
        layout(location = 0) in flat uint inParticleIndex;

        void main() {
            // Write the particle index into the visibility buffer.
            // The depth test is handled automatically by the rasterizer.
            outVisibilityID = inParticleIndex;
        }
    "#;
}

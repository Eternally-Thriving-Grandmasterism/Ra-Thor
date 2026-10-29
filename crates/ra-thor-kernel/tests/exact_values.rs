use ra_thor_kernel::{
    get_mercy_shader_wgsl, run_mercy_compute, MERCY_CLAMP_SOURCE_CEILING,
    MERCY_DECAY_SOURCE_LITERAL, MERCY_OFFSET_SOURCE_LITERAL,
};

fn trimmed_nonempty(source: &str) -> Vec<&str> {
    source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect()
}

#[test]
fn decay_coefficients_are_the_source_literals() {
    assert_eq!(MERCY_DECAY_SOURCE_LITERAL.to_bits(), 0.92f32.to_bits());
    assert_eq!(MERCY_OFFSET_SOURCE_LITERAL.to_bits(), 0.08f32.to_bits());
    assert_eq!(MERCY_CLAMP_SOURCE_CEILING.to_bits(), 4.0f32.to_bits());
}

#[test]
fn compute_applies_the_source_formula() {
    assert_eq!(run_mercy_compute(&[]), Vec::<f32>::new());

    let series = run_mercy_compute(&[9.0, 1.0, 2.0, 3.0, 8.0]);
    assert_eq!(series.len(), 5);
    assert_eq!(series[0].to_bits(), MERCY_OFFSET_SOURCE_LITERAL.to_bits());
    assert_eq!(series[4].to_bits(), MERCY_OFFSET_SOURCE_LITERAL.to_bits());
    assert_eq!(
        series[1].to_bits(),
        (1.0f32 * MERCY_DECAY_SOURCE_LITERAL + MERCY_OFFSET_SOURCE_LITERAL).to_bits()
    );
    assert_eq!(
        series[2].to_bits(),
        (2.0f32 * MERCY_DECAY_SOURCE_LITERAL + MERCY_OFFSET_SOURCE_LITERAL).to_bits()
    );
    assert_eq!(
        series[3].to_bits(),
        (3.0f32 * MERCY_DECAY_SOURCE_LITERAL + MERCY_OFFSET_SOURCE_LITERAL).to_bits()
    );

    let high = run_mercy_compute(&[5.0, 100.0]);
    assert_eq!(high.len(), 2);
    assert_eq!(high[0].to_bits(), series[0].to_bits());
    assert_eq!(high[1].to_bits(), MERCY_CLAMP_SOURCE_CEILING.to_bits());
    let unclamped = 100.0f32 * MERCY_DECAY_SOURCE_LITERAL + MERCY_OFFSET_SOURCE_LITERAL;
    assert_eq!(unclamped.to_bits(), (100.0f32 * 0.92 + 0.08).to_bits());
    assert_ne!(unclamped.to_bits(), high[1].to_bits());
}

#[test]
fn shader_text_matches_the_named_kind() {
    assert_eq!(get_mercy_shader_wgsl("nope"), "/* invalid shader */");
    assert_eq!(get_mercy_shader_wgsl("VERTEX"), "/* invalid shader */");
    assert_eq!(
        trimmed_nonempty(&get_mercy_shader_wgsl("vertex")),
        vec![
            "struct VertexInput {",
            "@builtin(instance_index) instance_index: u32,",
            "@location(0) position: vec3<f32>,",
            "@location(1) phase: f32,",
            "@location(2) mercy_intensity: f32,",
            "};",
            "struct VertexOutput {",
            "@builtin(position) clip_position: vec4<f32>,",
            "@location(0) mercy: f32,",
            "};",
            "@vertex",
            "fn main(in: VertexInput) -> VertexOutput {",
            "var out: VertexOutput;",
            "var pos = in.position + sin(in.phase + globals.time * 3.0) * 0.08 * in.mercy_intensity;",
            "out.clip_position = globals.projection * globals.view * vec4<f32>(pos, 1.0);",
            "out.mercy = in.mercy_intensity;",
            "return out;",
            "}",
        ]
    );
    assert_eq!(
        trimmed_nonempty(&get_mercy_shader_wgsl("fragment")),
        vec![
            "@fragment",
            "fn main(in: VertexOutput) -> @location(0) vec4<f32> {",
            "let mercy_color = mix(vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 0.85, 0.0), in.mercy);",
            "let lightning = sin(globals.time * 12.0 + in.mercy * 6.0) * 0.5 + 0.5;",
            "let nth_glow = sin(globals.time * 24.0) * 0.3 + 0.7;",
            "return vec4<f32>(mercy_color * (1.0 + lightning * 1.8 * nth_glow), 0.92);",
            "}",
        ]
    );
    assert_eq!(
        trimmed_nonempty(&get_mercy_shader_wgsl("compute")),
        vec![
            "@group(0) @binding(0) var<storage, read_write> roots: array<f32, 240>;",
            "@compute @workgroup_size(64)",
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
            "if (gid.x >= 240u) { return; }",
            "let mercy = roots[gid.x];",
            "roots[gid.x] = mercy * 0.92 + 0.08; // nilpotent mercy decay",
            "}",
        ]
    );
}

@group(0) @binding(0) var<uniform> ctx: Context;

struct Context {
    transform: mat4x4f,
    camera_direction: vec3f,
    color: vec3f
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(1) world_position: vec3f,
}

@vertex
fn vert(@location(0) position: vec4f) -> VertexOutput {
    return VertexOutput(ctx.transform * position, position.xyz);
}

@fragment
fn frag(in: VertexOutput) -> @location(0) vec4f {
    // Flat colors (orange, or red for the hole about to be removed) so drain
    // holes stand out from models and supports.
    let intensity = blinn_phong(screen_normal(in.world_position), ctx.camera_direction);
    return vec4(ctx.color * (0.35 + 0.65 * intensity), 1.0);
}

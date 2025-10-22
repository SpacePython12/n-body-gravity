struct CameraMatrices {
    lookat: mat4x4<f32>,
    projection: mat4x4<f32>
}

struct Body {
    pos: vec4<f32>,
    vel: vec4<f32>,
    acc: vec4<f32>,
    force: vec4<f32>,
    w_dir: vec4<f32>,
    p_dir: vec4<f32>,
    q_dir: vec4<f32>,
    mass: f32,
    radius: f32,
    ecc: f32,
    angm: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraMatrices;

@group(0) @binding(1)
var<storage, read> bodies: array<Body>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    const vertices = array(
        vec4( 1.0, -1.0, 0.0, 0.0),
        vec4(-1.0, -1.0, 0.0, 0.0),
        vec4( 1.0,  1.0, 0.0, 0.0),
        vec4(-1.0,  1.0, 0.0, 0.0),
    );
    const texcoords = array(
        vec2( 1.0, -1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0),
    );

    var output: VertexOutput;

    let body = bodies[instance_index];

    output.position = camera.projection * ((camera.lookat * vec4(body.pos.xyz, 1.0)) + vertices[vertex_index] * body.radius);

    output.texcoord = texcoords[vertex_index];

    return output;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    if length(in.texcoord) <= 1.0 {
        return vec4(0.25, 0.25, 0.25, 1.0);
    } else {
        return vec4(0.0);
    }
}




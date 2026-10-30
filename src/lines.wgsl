struct CameraMatrices {
    lookat: mat4x4<f32>,
    projection: mat4x4<f32>
}

struct Body {
    pos: vec4<f32>,
    mass: f32,
    radius: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraMatrices;

@group(0) @binding(1)
var<storage, read> bodies: array<Body>;

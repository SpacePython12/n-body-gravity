struct CameraMatrices {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    pos: vec4<f32>,
}

struct Body {
    sphere: vec4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraMatrices;

@group(0) @binding(1)
var<storage, read> bodies: array<Body>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texcoord: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) sphere: vec4<f32>,
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    const sqrt_3 = 1.7320508;
    const vertices = array(
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
        // vec2( 0.0,    -2.0),
        // vec2( sqrt_3,  1.0),
        // vec2(-sqrt_3,  1.0),
    );
    const frac_4_3_pi = 4.1887903;

    var output: VertexOutput;

    let body = bodies[instance_index];

    let focal_length = 1.0;
    let aspect_scale = vec2(camera.proj[0][0], camera.proj[1][1]);

    let sphere = body.sphere;
    let position = sphere.xyz;
    let radius = sphere.w;
    let view_pos = (camera.view * vec4(position, 1.0));
    let view_z = view_pos.z + radius;
    let clip = camera.proj * vec4(0.0, 0.0, view_z, 1.0);

    let corner = ellipse_corner(sphere, focal_length, vertex_index, camera.view);

    output.position = vec4(corner.position*aspect_scale*clip.w, clip.zw);

    output.texcoord = corner.texcoord;

    output.color = body.color;

    output.sphere = sphere;

    return output;
}

struct CornerInfo {
    position: vec2<f32>,
    texcoord: vec2<f32>,
}

fn ellipse_corner(sph: vec4<f32>, fl: f32, i: u32, m: mat4x4<f32>) -> CornerInfo {
    const corners = array(
        vec2( 1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2(-1.0, -1.0),
    );

    var info: CornerInfo;

    let o = (m*vec4(sph.xyz, 1.0));
    let r = sph.w;

    let A = r*r - dot(o.yz, o.yz);
    let C = r*r - dot(o.xz, o.xz);
    let F = (r*r - dot(o.xy, o.xy))*fl*fl;
    let B = 2.0*o.x*o.y;
    let D = 2.0*o.x*o.z*fl;
    let E = 2.0*o.y*o.z*fl;

    let det = B*B - 4.0*A*C;
    let x0 = (2.0*C*D - B*E) / det;
    let y0 = (2.0*A*E - B*D) / det;
    let p0 = vec2(x0, y0);
    let theta = 0.5*atan2(-B, C-A);

    let f0 = F + (D*x0 + E*y0) * 0.5;
    let f1 = 0.5 * (A + C);
    let f2 = 0.5 * sqrt((A - C)*(A - C) + B*B);

    let a = sqrt(-f0 / (f1 + f2));
    let b = sqrt(-f0 / (f1 - f2));

    let pl = corners[i] * vec2(a,b);

    let tsin = sin(theta);
    let tcos = cos(theta);
    let tmat = mat2x2(
        vec2(-tsin,  tcos),
        vec2( tcos,  tsin),
    );

    info.position = p0 + tmat * pl;
    info.texcoord = tmat * corners[i];

    return info;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(
    in: VertexOutput
) -> FragmentOutput {
    const pi = 3.1415927;
    const tau = 2.0*pi;
    var output: FragmentOutput;

    if length(in.texcoord) > 1.0 {
        // output.color = vec4(1.0, 0.0, 0.0, 1.0);
        // return output;
        discard;
    }

    let center = in.sphere.xyz;
    let radius = in.sphere.w;

    let x = -in.texcoord.x;
    let y = -in.texcoord.y;
    let z = sqrt(1.0 - dot(in.texcoord, in.texcoord));

    let trans_view = transpose(camera.view);
    let view_r = trans_view[0].xyz;
    let view_u = trans_view[1].xyz;
    let view_b = trans_view[2].xyz;

    let world_normal = mat3x3(
        view_r,
        view_u,
        view_b
    ) * vec3(x, y, z);

    let world_pos = world_normal * radius + center;

    let light_dir = normalize(vec3(1.0, 0.0, 0.0));
    let light_dot = saturate(dot(light_dir, world_normal));

    let light = vec3(1.0) * saturate(light_dot + 0.05);

    let clip_pos = camera.proj * camera.view * vec4(world_pos, 1.0);

    // let depth = sqrt(1.0 - dot(in.texcoord, in.texcoord)) + in.position.z;
    // let depth = in.position.z;
    let depth = clip_pos.z / clip_pos.w;

    // let angle = atan2(in.texcoord.y, in.texcoord.x) / tau;
    // let color = hue_to_rgb(angle) * dot(in.texcoord, in.texcoord);
    // let color = vec3(fract(angle), dot(in.texcoord, in.texcoord), 0.0);
    // let color = vec3(0.0, saturate(in.texcoord.x), saturate(in.texcoord.y));
    let color = in.color.rgb * light;

    output.color = vec4(color, in.color.a);
    output.depth = depth;

    return output;
}

fn hue_to_rgb(hue: f32) -> vec3<f32> {
    let ang = fract(hue);
    let r = abs(ang * 6 - 3) - 1;
    let g = 2 - abs(ang * 6 - 2);
    let b = 2 - abs(ang * 6 - 4);
    return saturate(vec3(r,g,b));
}

fn sph_intersect(origin: vec3<f32>, dir: vec3<f32>, sphere: vec4<f32>) -> f32 {
    let origin_to_center = origin - sphere.xyz;
    let b = dot(origin_to_center, dir);
    let c = dot(origin_to_center, origin_to_center) - sphere.w * sphere.w;
    let h = b*b - c;
    if h < 0.0 {
        return -1.0;
    } else {
        return -b - sqrt(h);
    }
}

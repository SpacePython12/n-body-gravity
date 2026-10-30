use std::{
    f32::consts::{FRAC_PI_2, PI, TAU},
    hash::{Hash, Hasher},
};

use math::*;

use crate::sim::Body;

pub mod math;
pub mod octree;
/// Octree implementation based on DeadLock's quadtree implementation: https://www.youtube.com/watch?v=nZHjD3cI-EU
// pub mod octree_old;
pub mod render;
pub mod sim;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraInfo {
    pub view: [f32; 16],
    pub proj: [f32; 16],
    pub pos: [f32; 4],
}

#[repr(C, align(0x10))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderBody {
    pub sphere: [f32; 4],
    pub color: [f32; 4],
}

fn boring_sim() -> sim::Simulation {
    let mut sim = sim::Simulation::new(1000.0, 0.0, 0.0, f32::INFINITY);

    sim.insert(Body::from_lins(
        vec3a(0.0, 0.0, 0.0),
        vec3a(0.0, 0.0, 0.0),
        10000.0,
        50.0,
    ));
    sim.insert(Body::from_lins(
        vec3a(198.3, 0.0, 26.11),
        // vec3a(0.0, 230.0, 0.0),
        vec3a(0.0, 0.0, 0.0),
        100.0,
        10.0,
    ));
    sim.insert(Body::from_lins(
        vec3a(448.3, 0.0, -39.22),
        // vec3a(0.0, 150.0, 0.0),
        vec3a(0.0, 0.0, 0.0),
        90.0,
        9.0,
    ));
    sim.insert(Body::from_lins(
        vec3a(700.0, 0.0, 0.0),
        // vec3a(0.0, 120.0, 0.0),
        vec3a(0.0, 0.0, 0.0),
        80.0,
        8.0,
    ));

    sim
}

fn interesting_sim() -> sim::Simulation {
    use rand::Rng;

    const N: usize = 5000;

    const BOUND_DIST: f32 = 10000.0;

    const MAX_DIST: f32 = 4000.0;
    const MIN_DIST: f32 = 500.0;
    const SPACING: f32 = (MAX_DIST - MIN_DIST) / (N as f32);
    const PERTURB: f32 = SPACING * 0.0;

    const MAX_MASS: f32 = 1000.0;
    const MIN_MASS: f32 = 500.0;
    const AVG_MASS: f32 = (MAX_MASS + MIN_MASS) * 0.5;

    const MIN_DENSITY: f32 = 0.9;
    const MAX_DENSITY: f32 = 1.1;

    const INCLINE: f32 = 70.0f32.to_radians();

    const CENTER_MASS: f32 = 10000000.0;

    const TOTAL_MASS: f32 = CENTER_MASS + AVG_MASS * N as f32;

    fn radius_from_mass_density(mass: f32, density: f32) -> f32 {
        ((3.0 * mass) / (2.0 * TAU * density)).cbrt()
    }

    let mut rng = rand::rng();

    let mut sim = sim::Simulation::new(10.0, 0.5, 0.01, BOUND_DIST);

    let center_radius =
        radius_from_mass_density(CENTER_MASS, rng.random_range(MIN_DENSITY..=MAX_DENSITY));

    sim.insert(Body::from_lins(
        Vec3A::ZERO,
        Vec3A::ZERO,
        CENTER_MASS,
        center_radius,
    ));

    for i in 0..N {
        let mass = rng.random_range(MIN_MASS..=MAX_MASS);
        let density = rng.random_range(MIN_DENSITY..=MAX_DENSITY);
        let radius = radius_from_mass_density(mass, density);

        let base_distance = MIN_DIST + SPACING * (i as f32);
        let perturb = rng.random_range(-PERTURB..=PERTURB);

        let distance = base_distance + perturb;
        let velocity = ((sim.octree.g * (CENTER_MASS + mass)) / distance).sqrt();

        let angle = rng.random_range(0.0..TAU);

        let (asin, acos) = angle.sin_cos();

        let inclination =
            Quat::from_axis_angle(vec3(-asin, acos, 0.0), rng.random_range(-INCLINE..=INCLINE));

        // let velocity = inclination.mul_vec3a(vec3a(velocity * -asin, velocity * acos, 0.0));
        let position = inclination.mul_vec3a(vec3a(distance * acos, distance * asin, 0.0));
        let velocity = vec3a(velocity * -asin, velocity * acos, 0.0);

        sim.insert(Body::from_lins(position, velocity, mass, radius));
    }

    sim
}

fn main() {
    let mut output_file = if std::fs::exists("metrics.csv").is_ok_and(std::convert::identity) {
        Some(std::fs::File::create("metrics.csv").expect("Unable to open file"))
    } else {
        None
    };
    render::run(|renderer| {
        let device = renderer.resources().device().clone();
        let queue = renderer.resources().queue().clone();

        let mut sim = interesting_sim();
        // let mut sim = boring_sim();
        let max_body_count = sim.indexes.len();
        let mut body_data = Vec::with_capacity(max_body_count);

        let projection_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size_of::<CameraInfo>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let bodies_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size_of::<RenderBody>() * max_body_count) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::include_wgsl!("billboard.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: renderer.resources().depth_texture_format(),
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: renderer.resources().surface_texture_format(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::OVER,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        projection_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        bodies_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        if let Some(output) = output_file.as_mut() {
            use std::io::Write;
            writeln!(output, "Time (s),Body Count,Angular Momentum (?)").unwrap();
        }

        const SQRT_3: f32 = 1.7320508;

        let mut total_time = 0.0f32;

        let mut last_record_time = 0.0f32;

        let mut camera_dist_exp = 8.0f32;
        let mut camera_yaw = 0.0;
        let mut camera_pitch = 0.0;

        let mut time = std::time::Instant::now();
        renderer.set_render_callback(move |renderer, ctx| {
            const RECORD_INTERVAL: f32 = 0.5;
            let dt = time.elapsed().as_secs_f32();
            time = std::time::Instant::now();
            total_time += dt;
            sim.step(dt);

            // if renderer.mouse_button_state(winit::event::MouseButton::Right) {
            //     sim.step(0.01);
            // }

            body_data.clear();
            for handle in sim.handles.iter() {
                let slot = handle.slot() as usize;
                let mut hasher = std::hash::DefaultHasher::new();
                handle.hash(&mut hasher);
                let hash = hasher.finish();
                let hash = (hash as u16)
                    ^ ((hash >> 16) as u16)
                    ^ ((hash >> 32) as u16)
                    ^ ((hash >> 48) as u16);
                let ang = (hash as f32) / (u16::MAX as f32);
                let r = (ang * 6.0 - 3.0).abs() - 1.0;
                let g = 2.0 - (ang * 6.0 - 2.0).abs();
                let b = 2.0 - (ang * 6.0 - 4.0).abs();

                body_data.push(RenderBody {
                    sphere: sim.positions[slot].extend(sim.radii[slot]).to_array(),
                    color: vec4(r, g, b, 1.0).to_array(),
                });
            }

            if total_time - last_record_time >= RECORD_INTERVAL {
                last_record_time += RECORD_INTERVAL;
                if let Some(output) = output_file.as_mut() {
                    use std::io::Write;
                    writeln!(
                        output,
                        "{:.2},{},{:.8e}",
                        total_time,
                        sim.indexes.len(),
                        sim.total_angular_momentum()
                    )
                    .unwrap();
                }
            }

            let camera_pos = {
                const CURSOR_SENSE_YAW: f32 = 0.01;
                const CURSOR_SENSE_PITCH: f32 = 0.01;
                const SCROLL_SENSE: f32 = 0.0625;
                if renderer.mouse_button_state(winit::event::MouseButton::Left) {
                    let cursor_delta = renderer.cursor_delta();
                    camera_yaw = (camera_yaw - cursor_delta.x * CURSOR_SENSE_YAW) % TAU;
                    camera_pitch = (camera_pitch + cursor_delta.y * CURSOR_SENSE_PITCH)
                        .clamp(-FRAC_PI_2 + f32::EPSILON, FRAC_PI_2 - f32::EPSILON);
                }
                let scroll_delta = renderer.scroll_delta();
                camera_dist_exp += scroll_delta.y * SCROLL_SENSE;

                let (z, xy) = camera_pitch.sin_cos();
                let (y, x) = camera_yaw.sin_cos();

                vec3a(x * xy, y * xy, z) * camera_dist_exp.exp2()
            };

            let win_dim = renderer.resources().surface_texture_size();
            let (win_width, win_height) = (win_dim.x, win_dim.y);

            let aspect_ratio = (win_width as f32) / (win_height as f32);

            let center = sim.center_of_mass;

            let camera_pos = (center + camera_pos).to_vec3();
            let view_matrix = Mat4::look_at_rh(camera_pos, center.into(), Vec3::Z);
            let proj_matrix =
                Mat4::perspective_infinite_rh(90.0f32.to_degrees(), aspect_ratio, 1.0);

            queue.write_buffer(
                &projection_buffer,
                0,
                bytemuck::bytes_of(&CameraInfo {
                    view: view_matrix.to_cols_array(),
                    proj: proj_matrix.to_cols_array(),
                    pos: camera_pos.extend(0.0).to_array(),
                }),
            );

            queue.write_buffer(
                &bodies_buffer,
                0,
                bytemuck::cast_slice(body_data.as_slice()),
            );

            queue.submit(None);

            {
                let mut render_pass =
                    ctx.command_encoder
                        .begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: None,
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: ctx.surface_texture_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 0.0,
                                        g: 0.0,
                                        b: 0.0,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: ctx.depth_texture_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                        });

                render_pass.set_pipeline(&pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.draw(0..4, 0..body_data.len() as u32);
            }
            Ok(())
        });
        Ok(())
    });
}

#[test]
fn test_sim() {
    let mut sim = boring_sim();

    let mut time;
    sim.step(0.0);
    sim.rebuild_octree();
    for i in 0..5 {
        let position = sim
            .handles
            .iter()
            .map(|handle| sim.positions[handle.slot() as usize])
            .collect::<Vec<_>>();

        let velocity = sim
            .handles
            .iter()
            .map(|handle| sim.linvels[handle.slot() as usize])
            .collect::<Vec<_>>();

        println!("--- Begin accel calcs ---");
        let accel = sim
            .handles
            .iter()
            .map(|handle| sim.octree.accel_at(sim.positions[handle.slot() as usize]))
            .collect::<Vec<_>>();
        println!("--- End accel calcs ---");

        println!("Step {i}:");
        println!("P: {:?}", position);
        println!("V: {:?}", velocity);
        println!("A: {:?}", accel);

        time = std::time::Instant::now();
        sim.step(0.1);
        println!("v--- took {} seconds", time.elapsed().as_secs_f32());
    }
}

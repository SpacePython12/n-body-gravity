use std::f32::consts::{FRAC_PI_2, PI, TAU};

use glam::*;
use rand::Rng;

pub mod sim;
pub mod render;
/// Octree implementation based on DeadLock's quadtree implementation: https://www.youtube.com/watch?v=nZHjD3cI-EU
pub mod octree;


fn interesting_sim() -> sim::Simulation {
    const N: usize = 2000;

    const MAX_DIST: f32 = 4000.0;
    const MIN_DIST: f32 = 400.0;
    const SPACING: f32 = (MAX_DIST - MIN_DIST) / (N as f32);
    const PERTURB: f32 = 50.0;

    const MAX_MASS: f32 = 1000.0;
    const MIN_MASS: f32 = 200.0;

    const MIN_DENSITY: f32 = 0.9;
    const MAX_DENSITY: f32 = 1.0;

    const INCLINE: f32 = PI / 24.0;

    const CENTER_MASS: f32 = 10000000.0;

    fn radius_from_mass_density(mass: f32, density: f32) -> f32 {
        ((3.0 * mass)/(2.0 * TAU * density)).cbrt()
    }

    let mut rng = rand::rng();

    let mut sim = sim::Simulation::with_capacity(10.0, N+1);

    let center_radius = radius_from_mass_density(CENTER_MASS, rng.random_range(MIN_DENSITY..=MAX_DENSITY));

    sim.insert(center_radius, CENTER_MASS, Vec3A::ZERO, Vec3A::ZERO);


    for i in 0..N {
        let mass = rng.random_range(MIN_MASS..=MAX_MASS);
        let density = rng.random_range(MIN_DENSITY..=MAX_DENSITY);
        let radius = radius_from_mass_density(mass, density);

        let base_distance = MIN_DIST + SPACING * (i as f32);
        let perturb = rng.random_range(-PERTURB..=PERTURB);

        let distance = base_distance + perturb;
        let velocity = ((sim.grav_const * (CENTER_MASS + mass)) / distance).sqrt();

        let angle = rng.random_range(0.0..TAU);

        let (asin, acos) = angle.sin_cos();
        
        let position = vec3a(distance * acos, distance * asin, 0.0);

        let inclination = Quat::from_axis_angle(vec3(acos, asin, 0.0), rng.random_range(-INCLINE..=INCLINE));

        sim.insert(radius, mass, position, inclination.mul_vec3a(vec3a(velocity * -asin, velocity * acos, 0.0)));
    }

    sim
}

fn main() {
    render::run(|renderer| {
        let device = renderer.resources().device().clone();
        let queue = renderer.resources().queue().clone();

        let mut sim = interesting_sim();

        let projection_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (size_of::<Mat4>() * 2) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let bodies_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of_val(sim.bodies.as_slice()) as u64,
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
                buffers: &[]
            }, 
            primitive: wgpu::PrimitiveState { 
                topology: wgpu::PrimitiveTopology::TriangleStrip, 
                strip_index_format: None, 
                cull_mode: None,
                ..Default::default()
            }, 
            depth_stencil: Some(wgpu::DepthStencilState { 
                format: renderer.resources().depth_texture_format(), 
                depth_write_enabled: true, 
                depth_compare: wgpu::CompareFunction::Always, 
                stencil: Default::default(), 
                bias: Default::default() 
            }), 
            multisample: Default::default(), 
            fragment: Some(wgpu::FragmentState { 
                module: &module, 
                entry_point: Some("fs_main"), 
                compilation_options: Default::default(), 
                targets: &[
                    Some(wgpu::ColorTargetState { 
                        format: renderer.resources().surface_texture_format(), 
                        blend: Some(wgpu::BlendState { 
                            color: wgpu::BlendComponent::OVER, 
                            alpha: wgpu::BlendComponent::REPLACE,
                        }), 
                        write_mask: wgpu::ColorWrites::COLOR
                    })
                ] 
            }), 
            multiview: None, 
            cache: None 
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { 
            label: None, 
            layout: &pipeline.get_bind_group_layout(0), 
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(projection_buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(bodies_buffer.as_entire_buffer_binding()),
                },
            ] 
        });
        let mut total_time = 0.0f32;

        let mut camera_dist_exp = 11.0f32;
        let mut camera_yaw = 0.0;
        let mut camera_pitch = 0.0;

        let mut time = std::time::Instant::now();
        renderer.set_render_callback(move |renderer, ctx| {
            let dt = time.elapsed().as_secs_f32();
            total_time += dt;
            sim.step(dt, 0.1);

            let camera_pos = {
                const CURSOR_SENSE_YAW: f32 = 1.0;
                const CURSOR_SENSE_PITCH: f32 = 1.0;
                const SCROLL_SENSE: f32 = 0.25;
                if renderer.mouse_button_state(winit::event::MouseButton::Left) {
                    
                    let cursor_delta = renderer.cursor_delta() * dt;
                    camera_yaw = (camera_yaw + cursor_delta.x * CURSOR_SENSE_YAW) % TAU;
                    camera_pitch = (camera_pitch - cursor_delta.y * CURSOR_SENSE_PITCH).clamp(-FRAC_PI_2+f32::EPSILON, FRAC_PI_2-f32::EPSILON);
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

            let center = sim.cog_pos;

            let lookat_matrix = Mat4::look_at_rh((center + camera_pos).into(), center.into(), Vec3::Z);
            let projection_matrix = Mat4::perspective_infinite_rh(90.0f32.to_degrees(), aspect_ratio, 1.0);
            
            queue.write_buffer(&projection_buffer, 0, bytemuck::bytes_of(&[lookat_matrix, projection_matrix]));

            queue.write_buffer(&bodies_buffer, 0, bytemuck::cast_slice(sim.bodies.as_slice()));

            queue.submit(None);

            {
                let mut render_pass = ctx.command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor { 
                    label: None, 
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment { 
                            view: ctx.surface_texture_view, 
                            depth_slice: None, 
                            resolve_target: None, 
                            ops: wgpu::Operations { 
                                load: wgpu::LoadOp::Clear(wgpu::Color { 
                                    r: 0.05, 
                                    g: 0.05, 
                                    b: 0.05, 
                                    a: 1.0 
                                }), 
                                store: wgpu::StoreOp::Store
                            }
                        })
                    ], 
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { 
                        view: ctx.depth_texture_view, 
                        depth_ops: Some(wgpu::Operations { 
                            load: wgpu::LoadOp::Clear(1.0), 
                            store: wgpu::StoreOp::Store
                        }), 
                        stencil_ops: None 
                    }), 
                    timestamp_writes: None, 
                    occlusion_query_set: None 
                });

                render_pass.set_pipeline(&pipeline);
                render_pass.set_bind_group(0, &bind_group, &[]);
                render_pass.draw(0..4, 0..sim.bodies.len() as u32);
            }

            time = std::time::Instant::now();
            Ok(())
        });
        Ok(())
    });
}

// #[test]
// fn test_sim() {
//     let mut sim = sim::Simulation::with_capacity(4);

//     sim.insert(10000.0, vec3a(0.0, 0.0, 0.0), vec3a(0.0, 0.0, 0.0));
//     sim.insert(100.0, vec3a(198.3, 0.0, 26.11), vec3a(0.0, 230.0, 0.0));
//     sim.insert(90.0, vec3a(448.3, 0.0, -39.22), vec3a(0.0, 150.0, 0.0));
//     sim.insert(80.0, vec3a(700.0, 0.0, 0.0), vec3a(0.0, 120.0, 0.0));

//     let mut time;
//     sim.step(0.0, 0.0);    
//     for i in 0..3 {
//         println!("Step {i}:");
//         println!("{:?}", sim.position);
//         println!("{:?}", sim.velocity);
//         println!("{:?}", sim.force);
//         println!("{:?}", sim.accel);

//         time = std::time::Instant::now();
//         sim.step(0.01, 0.01);
//         println!("v--- took {} seconds", time.elapsed().as_secs_f32());
//     }
// }

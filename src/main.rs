use std::{
    f32::consts::{FRAC_PI_2, PI, TAU},
    hash::{Hash, Hasher},
    sync::{Arc, atomic::AtomicBool},
    thread::JoinHandle,
};

use eframe::{egui_wgpu, wgpu};
use math::*;
use parking_lot::{Condvar, Mutex, MutexGuard};
use rand::RngExt;
use rand_distr::Distribution;

use crate::sim::Body;

pub mod math;
pub mod octree;
/// Octree implementation based on DeadLock's quadtree implementation: https://www.youtube.com/watch?v=nZHjD3cI-EU
// pub mod octree_old;
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

fn cloud_collapse_sim() -> sim::Simulation {
    use rand::Rng;

    const N: usize = 5000;

    const BOUND_DIST: f32 = 5000.0;

    const MAX_DIST: f32 = 2000.0;
    const MIN_DIST: f32 = 500.0;

    const AVG_MASS: f32 = 500.0;
    const STD_MASS: f32 = 10.0;

    const AVG_DENSITY: f32 = 1.0;
    const STD_DENSITY: f32 = 0.1;

    const VEL_ALPHA: f32 = 0.05;

    const G: f32 = 10.0;

    fn radius_from_mass_density(mass: f32, density: f32) -> f32 {
        ((3.0 * mass) / (2.0 * TAU * density)).cbrt()
    }

    let mut rng = rand::rng();

    let mut sim = sim::Simulation::new(G, 0.5, 0.01, BOUND_DIST);

    let unit_sphere = rand_distr::UnitSphere;
    let vel_normal = rand_distr::Normal::new(
        0.0,
        VEL_ALPHA * G * (AVG_MASS * N as f32) / (5.0 * MAX_DIST),
    )
    .unwrap();
    let dist_uniform = rand_distr::Uniform::new_inclusive(MIN_DIST, MAX_DIST).unwrap();
    let mass_normal = rand_distr::Normal::new(AVG_MASS, STD_MASS).unwrap();
    let density_normal = rand_distr::Normal::new(AVG_DENSITY, STD_DENSITY).unwrap();

    for i in 0..N {
        let mass = mass_normal.sample(&mut rng);
        let density = density_normal.sample(&mut rng);
        let radius = ((3.0 * mass) / (2.0 * TAU * density)).cbrt();

        let pos_dir = Vec3A::from_array(unit_sphere.sample(&mut rng));
        let position = pos_dir * dist_uniform.sample(&mut rng);

        let vel_dir = Vec3A::from_array(unit_sphere.sample(&mut rng))
            .cross(pos_dir)
            .normalize_or_zero();
        let velocity = vel_dir * vel_normal.sample(&mut rng);

        sim.insert(Body::from_lins(position, velocity, mass, radius));
    }

    sim
}

fn interesting_sim() -> sim::Simulation {
    use rand::Rng;

    const N: usize = 5000;

    const BOUND_DIST: f32 = 10000.0;

    const MAX_DIST: f32 = 3000.0;
    const MIN_DIST: f32 = 500.0;
    const SPACING: f32 = (MAX_DIST - MIN_DIST) / (N as f32);
    const PERTURB: f32 = SPACING * 0.0;

    const MAX_MASS: f32 = 1000.0;
    const MIN_MASS: f32 = 500.0;
    const AVG_MASS: f32 = (MAX_MASS + MIN_MASS) * 0.5;

    const MIN_DENSITY: f32 = 0.9;
    const MAX_DENSITY: f32 = 1.1;

    const INCLINE: f32 = 50.0f32.to_radians();

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

struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub dist_exp: f32,
    pub fov: f32,
}

impl OrbitCamera {
    pub fn to_camera_info(&self, center: Vec3A, aspect_ratio: f32) -> CameraInfo {
        let (z, xy) = self.pitch.sin_cos();
        let (y, x) = self.yaw.sin_cos();

        let look_dir = -vec3a(x * xy, y * xy, z);
        let camera_pos = -look_dir * self.dist_exp.exp2() + center;
        let up_dir = look_dir.cross(Vec3A::Z).normalize().cross(look_dir);
        let rolled_up_dir = Quat::from_axis_angle(look_dir.to_vec3(), self.roll).mul_vec3a(up_dir);

        let view_matrix = Mat4::look_to_rh(
            camera_pos.to_vec3(),
            look_dir.to_vec3(),
            rolled_up_dir.to_vec3(),
        );
        let proj_matrix = Mat4::perspective_infinite_rh(self.fov, aspect_ratio, 1.0);

        CameraInfo {
            view: view_matrix.to_cols_array(),
            proj: proj_matrix.to_cols_array(),
            pos: camera_pos.extend(1.0).to_array(),
        }
    }
}

struct BodyRenderResources {
    projection_buffer: wgpu::Buffer,
    bodies_buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

struct BodyRenderCallback {
    body_data: Vec<RenderBody>,
    camera_info: CameraInfo,
}

impl egui_wgpu::CallbackTrait for BodyRenderCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources = callback_resources.get::<BodyRenderResources>().unwrap();
        queue.write_buffer(
            &resources.projection_buffer,
            0,
            bytemuck::bytes_of(&self.camera_info),
        );

        queue.write_buffer(
            &resources.bodies_buffer,
            0,
            bytemuck::cast_slice(self.body_data.as_slice()),
        );

        queue.submit(None);

        vec![]
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources = callback_resources.get::<BodyRenderResources>().unwrap();
        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.bind_group, &[]);
        render_pass.draw(0..4, 0..self.body_data.len() as u32);
    }
}

struct SimulationApp {
    sim: Arc<Mutex<sim::Simulation>>,
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    notifier: Arc<Condvar>,
    camera: OrbitCamera,
    tick_thread: JoinHandle<anyhow::Result<()>>,
}

impl SimulationApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc.wgpu_render_state.as_ref().expect("not using wgpu");
        let device = &render_state.device;
        let queue = &render_state.queue;

        let mut sim = interesting_sim();
        // let mut sim = boring_sim();
        // let mut sim = cloud_collapse_sim();
        let max_body_count = sim.indexes.len();

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
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: render_state.target_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::OVER,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            multiview_mask: None,
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

        render_state
            .renderer
            .write()
            .callback_resources
            .insert(BodyRenderResources {
                projection_buffer,
                bodies_buffer,
                pipeline,
                bind_group,
            });

        let camera = OrbitCamera {
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            dist_exp: 8.0,
            fov: 90.0f32.to_radians(),
        };

        let paused = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(true));
        let notifier = Arc::new(Condvar::new());

        let sim = Arc::new(Mutex::new(sim));

        let dt = std::time::Duration::from_secs_f32(60.0f32.recip());

        let tick_thread = {
            let sim = sim.clone();
            let paused = paused.clone();
            let running = running.clone();
            let notifier = notifier.clone();
            std::thread::spawn(move || {
                let mut start = std::time::Instant::now();
                let mut sim_lock = sim.lock();
                while running.load(std::sync::atomic::Ordering::Relaxed) {
                    if !paused.load(std::sync::atomic::Ordering::Relaxed) {
                        sim_lock.step(dt.as_secs_f32());
                    }

                    notifier.wait_until(&mut sim_lock, start + dt);
                    start = std::time::Instant::now();
                }

                Ok(())
            })
        };

        Self {
            sim,
            camera,
            paused,
            running,
            notifier,
            tick_thread,
        }
    }
}

impl eframe::App for SimulationApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        const RECORD_INTERVAL: f32 = 0.5;
        let sim_lock = self.sim.lock();

        let mut body_data = Vec::new();
        for handle in sim_lock.handles.iter() {
            fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
                let sector = ((h % 1.0) * 6.0);
                let c = v * s;
                let x = c * (1.0 - ((sector % 2.0) - 1.0).abs());
                let m = v - c;
                match sector as u8 {
                    0 => vec3(c, x, 0.0) * m,
                    1 => vec3(x, c, 0.0) * m,
                    2 => vec3(0.0, c, x) * m,
                    3 => vec3(0.0, x, c) * m,
                    4 => vec3(x, 0.0, c) * m,
                    5 => vec3(c, 0.0, x) * m,
                    _ => unreachable!(),
                }
            }

            let slot = handle.slot() as usize;
            let mut hasher = std::hash::DefaultHasher::new();
            handle.hash(&mut hasher);
            let hash = hasher.finish();
            let hash = (hash as u16)
                ^ ((hash >> 16) as u16)
                ^ ((hash >> 32) as u16)
                ^ ((hash >> 48) as u16);
            let ang = (hash as f32) / (u16::MAX as f32);
            let density = sim_lock.masses[slot] / (sim_lock.radii[slot].powi(3) * PI * 4.0 / 3.0);
            let color = hsv_to_rgb(ang, density / (density + 3.0), 2.0);

            body_data.push(RenderBody {
                sphere: sim_lock.positions[slot]
                    .extend(sim_lock.radii[slot])
                    .to_array(),
                color: color.extend(1.0).to_array(),
            });
        }

        egui::CentralPanel::default().show(ui, |ui| {
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                const CURSOR_SENSE_YAW: f32 = 0.01;
                const CURSOR_SENSE_PITCH: f32 = 0.01;
                const SCROLL_SENSE: f32 = 0.0625;

                let (rect, response) = ui
                    .allocate_at_least(ui.available_size(), egui::Sense::DRAG | egui::Sense::HOVER);

                self.camera.yaw =
                    (self.camera.yaw + response.drag_delta().x * CURSOR_SENSE_YAW) % TAU;
                self.camera.pitch = (self.camera.pitch
                    - response.drag_delta().y * CURSOR_SENSE_PITCH)
                    .clamp(-FRAC_PI_2 + f32::EPSILON, FRAC_PI_2 - f32::EPSILON);
                if response.hovered() {
                    self.camera.dist_exp +=
                        ui.input(|input| input.smooth_scroll_delta().y) * SCROLL_SENSE;
                }

                let aspect_ratio = rect.width() / rect.height();

                let center = sim_lock.center_of_mass;

                let camera_info = self.camera.to_camera_info(center, aspect_ratio);

                ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                    rect,
                    BodyRenderCallback {
                        body_data,
                        camera_info,
                    },
                ));
            });
        });

        ui.ctx().request_repaint();
    }
}

fn main() -> anyhow::Result<()> {
    let native_options = eframe::NativeOptions {
        depth_buffer: 32,
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "newtonian-gravity",
        native_options,
        Box::new(|cc| Ok(Box::new(SimulationApp::new(cc)))),
    )?;

    Ok(())
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

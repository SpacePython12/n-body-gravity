use glam::*;

use rayon::prelude::*;

use crate::octree::Octree;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimBody {
    pub pos: Vec3A,
    pub vel: Vec3A,
    pub acc: Vec3A,
    pub force: Vec3A,
    pub w_dir: Vec3A,
    pub p_dir: Vec3A,
    pub q_dir: Vec3A,
    pub mass: f32,
    pub radius: f32,
    pub angm: f32,
    pub ecc: f32,
}

pub struct Simulation {
    pub grav_const: f32,

    pub bodies: Vec<SimBody>,
    pub octree: Octree,

    pub cog_mass: f32,
    pub cog_pos: Vec3A,
    pub cog_vel: Vec3A,
}

impl Simulation {

    pub fn new(grav_const: f32) -> Self {
        Self {
            grav_const,
            bodies: Vec::new(),
            octree: Octree::new(1.0, 1.0, 16, 4096),

            cog_mass: 0.0,
            cog_pos: Vec3A::ZERO,
            cog_vel: Vec3A::ZERO,
        }
    }

    pub fn with_capacity(grav_const: f32, capacity: usize) -> Self {
        Self {
            bodies: Vec::with_capacity(capacity),
            ..Self::new(grav_const)
        }
    }

    pub fn insert(&mut self, radius: f32, mass: f32, position: Vec3A, velocity: Vec3A) {

        self.bodies.push(unsafe {
            let mut value: std::mem::MaybeUninit<SimBody> = std::mem::MaybeUninit::uninit();

            (*value.assume_init_mut()).pos = position;
            (*value.assume_init_mut()).vel = velocity;
            (*value.assume_init_mut()).mass = mass;
            (*value.assume_init_mut()).radius = radius;

            value.assume_init()
        });
    }

    fn remove(&mut self, index: usize) -> bool {
        if self.bodies.is_empty() { return false; }
        match index.cmp(&(self.bodies.len()-1)) {
            std::cmp::Ordering::Less => {
                self.bodies.swap_remove(index);
                true
            },
            std::cmp::Ordering::Equal => {
                self.bodies.pop();
                true
            },
            std::cmp::Ordering::Greater => false,
        }
    }

    fn combine(&mut self, greater: usize, lesser: usize) {
        const FRAC_4_3_PI: f32 = 4.0 * std::f32::consts::FRAC_PI_3;

        let greater_body = &self.bodies[greater];
        let lesser_body = &self.bodies[lesser];

        let greater_mass = greater_body.mass;
        let lesser_mass = lesser_body.mass;

        let final_mass = greater_mass + lesser_mass;

        let greater_velocity = greater_body.vel;
        let lesser_velocity = lesser_body.vel;

        let greater_momentum = greater_mass * greater_velocity;
        let lesser_momentum = lesser_mass * lesser_velocity;

        let final_momentum = greater_momentum + lesser_momentum;
        let final_velocity = final_momentum / final_mass;

        let greater_radius = greater_body.radius;
        let lesser_radius = lesser_body.radius;

        let greater_volume = FRAC_4_3_PI * greater_radius * greater_radius * greater_radius;
        let lesser_volume = FRAC_4_3_PI * lesser_radius * lesser_radius * lesser_radius;

        let greater_density = greater_mass / greater_volume;
        let lesser_density = lesser_mass / lesser_volume;

        let final_density = (greater_mass * greater_density + lesser_mass * lesser_density) / final_mass;
        let final_volume = final_mass / final_density;
        let final_radius = (final_volume / FRAC_4_3_PI).cbrt();

        let greater_body = &mut self.bodies[greater];

        greater_body.mass = final_mass;
        greater_body.radius = final_radius;
        greater_body.vel = final_velocity;

        self.remove(lesser);
        println!("Body collision! There are now {} bodies", self.bodies.len());
    }

    fn calc_forces_accels(&mut self) {
        // let mut row = 0usize;
        // for body in self.bodies.iter_mut() {
        //     body.force = Vec3A::ZERO;
        // }
        // while row < self.bodies.len()-1 {
        //     let mut col = row+1;
        //     while col < self.bodies.len() {
        //         let (disp, dist) = (self.bodies[col].position - self.bodies[row].position).normalize_and_length();

        //         let force = (self.grav_const * self.bodies[row].mass * self.bodies[col].mass * disp) / (dist * dist);

        //         // Every action has an equal and opposite reaction. - Isaac Newton, my goat
        //         self.bodies[row].force += force;
        //         self.bodies[col].force -= force;

        //         col += 1;
        //     }
        //     row += 1;
        // }
        // for body in self.bodies.iter_mut() {
        //     body.accel = body.force / body.mass;
        // }

        self.octree.build(&mut self.bodies);

        // for i in 0..self.bodies.len() {
        //     let acc = self.octree.force_at(&self.bodies, self.bodies[i].pos, self.bodies[i].mass, self.grav_const) / self.bodies[i].mass;
        //     self.bodies[i].acc = acc;
        // }

        self.octree.force(&mut self.bodies, self.grav_const);
    }

    fn calc_cog_state(&mut self) {
        self.cog_mass = 0.0;
        self.cog_pos = Vec3A::ZERO;
        self.cog_vel = Vec3A::ZERO;

        // let (
        //     cog_mass,
        //     cog_pos,
        //     cog_vel,
        // ) = self.bodies.par_iter().fold(
        //     || (0.0, Vec3A::ZERO, Vec3A::ZERO),
        //      |(cog_mass, cog_pos, cog_vel), body| {
        //         (
        //             cog_mass + body.mass,
        //             cog_pos + body.pos * body.mass,
        //             cog_vel + body.vel * body.mass,
        //         )
        //     }
        // ).reduce(
        //     || (0.0, Vec3A::ZERO, Vec3A::ZERO),
        //     |(mass_a, pos_a, vel_a), (mass_b, pos_b, vel_b)| {
        //         (mass_a + mass_b, pos_a + pos_b, vel_a + vel_b)
        //     }
        // );

        // self.cog_mass = cog_mass;
        // self.cog_pos = cog_pos / cog_mass;
        // self.cog_vel = cog_vel / cog_mass;

        // for body in self.bodies.iter() {
        //     self.cog_mass += body.mass;
        //     self.cog_pos += body.pos * body.mass;
        //     self.cog_vel += body.vel * body.mass;
        // }

        // self.cog_pos /= self.cog_mass;
        // self.cog_vel /= self.cog_mass;
    }

    fn leapfrog_integrate(&mut self, dt: f32) {
        const Y0: f32 = -1.70241438392f32;
        const Y1: f32 =  1.35120719196f32;
        const Y2: f32 =  0.67560359598f32;
        const Y3: f32 = -0.17560359598f32;

        if dt == 0.0 { return; }

        self.bodies.par_iter_mut().for_each(|body| {
            // Step 1a: r1 = r0 + c2*v0*dt
            body.pos += body.vel * Y2 * dt;
        });

        self.calc_forces_accels();
        self.bodies.par_iter_mut().for_each(|body| {
            // Step 1b: v1 = v0 + c1*a(r1)*dt
            body.vel += body.acc * Y1 * dt;

            // Step 2a: r2 = r1 + c3*v1*dt
            body.pos += body.vel * Y3 * dt;
        });

        self.calc_forces_accels();
        self.bodies.par_iter_mut().for_each(|body| {
            // Step 2b: v2 = v1 + c0*a(r2)*dt
            body.vel += body.acc * Y0 * dt;

            // Step 3a: r3 = r2 + c3*v2*dt
            body.pos += body.vel * Y3 * dt;
        });

        self.calc_forces_accels();
        self.bodies.par_iter_mut().for_each(|body| {
            // Step 3b: v3 = v2 + c1*a(r3)*dt
            body.vel += body.acc * Y1 * dt;

            // Step 4a: r4 = r3 + c2*v3*dt
            body.pos += body.vel * Y2 * dt;
        });

        // for body in self.bodies.iter_mut() {
        //     // Step 1a: r1 = r0 + c2*v0*dt
        //     body.pos += body.vel * Y2 * dt;
        // }

        // self.calc_forces_accels();
        // for body in self.bodies.iter_mut() {
        //     // Step 1b: v1 = v0 + c1*a(r1)*dt
        //     body.vel += body.acc * Y1 * dt;

        //     // Step 2a: r2 = r1 + c3*v1*dt
        //     body.pos += body.vel * Y3 * dt;
        // }

        // self.calc_forces_accels();
        // for body in self.bodies.iter_mut() {
        //     // Step 2b: v2 = v1 + c0*a(r2)*dt
        //     body.vel += body.acc * Y0 * dt;

        //     // Step 3a: r3 = r2 + c3*v2*dt
        //     body.pos += body.vel * Y3 * dt;
        // }

        // self.calc_forces_accels();
        // for body in self.bodies.iter_mut() {
        //     // Step 3b: v3 = v2 + c1*a(r3)*dt
        //     body.vel += body.acc * Y1 * dt;

        //     // Step 4a: r4 = r3 + c2*v3*dt
        //     body.pos += body.vel * Y2 * dt;
        // }
    }

    fn resolve_collisions(&mut self) {
        let mut i = 0usize;
        while i < self.bodies.len() {
            let mut max_mass = self.bodies[i].mass;
            let mut max_mass_index = None;
            let mut j = 0usize;
            while j < self.bodies.len() {
                if i != j && self.bodies[i].pos.distance(self.bodies[j].pos) <= (self.bodies[i].radius+self.bodies[j].radius) && self.bodies[j].mass >= max_mass {
                    max_mass = self.bodies[j].mass;
                    max_mass_index.replace(j);
                }
                j += 1;
            }

            if let Some(j) = max_mass_index {
                self.combine(j, i);
                continue; // Body i has been removed so dont skip ahead
            }

            i += 1;
        }
    }

    pub fn step(&mut self, mut dt: f32, max_step: f32) {
        // while dt > max_step {
        //     self.leapfrog_integrate(max_step);
        //     dt -= max_step;
        // }
        self.leapfrog_integrate(dt);

        self.resolve_collisions();

        self.calc_forces_accels();

        self.calc_cog_state();
    }
}
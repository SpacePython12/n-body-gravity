use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Mul;

use rayon::prelude::*;
use thunderdome::{Arena, Index};

use crate::math::*;

use crate::octree::Octree;

pub struct Bodies {
    arena: Arena<()>,
    positions: Vec<Vec3A>,
    linvels: Vec<Vec3A>,
    angvels: Vec<Vec3A>,
    masses: Vec<f32>,
    radii: Vec<f32>,
}

pub struct BodyRef<'a> {
    position: &'a Vec3A,
    linvel: &'a Vec3A,
    angvel: &'a Vec3A,
    mass: &'a f32,
    radius: &'a f32,
}

pub struct BodyMut<'a> {
    position: &'a mut Vec3A,
    linvel: &'a mut Vec3A,
    angvel: &'a mut Vec3A,
    mass: &'a mut f32,
    radius: &'a mut f32,
}

impl Bodies {
    pub const fn new() -> Self {
        Self {
            arena: Arena::new(),
            positions: Vec::new(),
            linvels: Vec::new(),
            angvels: Vec::new(),
            masses: Vec::new(),
            radii: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: Arena::with_capacity(capacity),
            positions: Vec::with_capacity(capacity),
            linvels: Vec::with_capacity(capacity),
            angvels: Vec::with_capacity(capacity),
            masses: Vec::with_capacity(capacity),
            radii: Vec::with_capacity(capacity),
        }
    }

    pub const fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    pub const fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    pub fn insert(&mut self, value: Body) -> Index {
        let index = self.arena.insert(());

        let len = self.positions.len();
        let slot = index.slot() as usize;

        if slot < len {
            self.positions[slot] = value.position;
            self.linvels[slot] = value.linvel;
            self.angvels[slot] = value.angvel;
            self.masses[slot] = value.mass;
            self.radii[slot] = value.radius;
        } else {
            if slot > len {
                self.positions.resize(slot, Vec3A::ZERO);
                self.linvels.resize(slot, Vec3A::ZERO);
                self.angvels.resize(slot, Vec3A::ZERO);
                self.masses.resize(slot, 0.0);
                self.radii.resize(slot, 0.0);
            }
            self.positions.push(value.position);
            self.linvels.push(value.linvel);
            self.angvels.push(value.angvel);
            self.masses.push(value.mass);
            self.radii.push(value.radius);
        }

        index
    }

    pub fn remove(&mut self, index: Index) -> Option<Body> {
        if self.arena.remove(index).is_some() {
            let slot = index.slot() as usize;
            Some(Body {
                position: self.positions[slot],
                linvel: self.linvels[slot],
                angvel: self.angvels[slot],
                mass: self.masses[slot],
                radius: self.radii[slot],
            })
        } else {
            None
        }
    }

    pub fn contains(&self, index: Index) -> bool {
        self.arena.contains(index)
    }

    pub fn get(&self, index: Index) -> Option<BodyRef<'_>> {
        if self.arena.contains(index) {
            let slot = index.slot() as usize;
            Some(BodyRef {
                position: &self.positions[slot],
                linvel: &self.linvels[slot],
                angvel: &self.angvels[slot],
                mass: &self.masses[slot],
                radius: &self.radii[slot],
            })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: Index) -> Option<BodyMut<'_>> {
        if self.arena.contains(index) {
            let slot = index.slot() as usize;
            Some(BodyMut {
                position: &mut self.positions[slot],
                linvel: &mut self.linvels[slot],
                angvel: &mut self.angvels[slot],
                mass: &mut self.masses[slot],
                radius: &mut self.radii[slot],
            })
        } else {
            None
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Body {
    position: Vec3A,
    linvel: Vec3A,
    angvel: Vec3A,
    mass: f32,
    radius: f32,
}

impl Body {
    pub fn from_lins(position: Vec3A, velocity: Vec3A, mass: f32, radius: f32) -> Self {
        Self {
            position,
            linvel: velocity,
            angvel: Vec3A::ZERO,
            mass,
            radius,
        }
    }

    pub fn mass(&self) -> f32 {
        self.mass
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn volume(&self) -> f32 {
        use std::f32::consts::PI;
        self.radius * self.radius * self.radius * PI * 4.0 / 3.0
    }

    pub fn inertia(&self) -> f32 {
        self.mass * self.radius * self.radius * 2.0 / 5.0
    }

    pub fn position(&self) -> Vec3A {
        self.position
    }

    pub fn linvel(&self) -> Vec3A {
        self.linvel
    }

    pub fn angvel(&self) -> Vec3A {
        self.angvel
    }

    fn integrate_pos(&mut self, dt: f32) {
        self.position = self.linvel.mul_add(Vec3A::splat(dt), self.position)
    }

    fn integrate_linvel(&mut self, accel: Vec3A, dt: f32) {
        self.linvel = accel.mul_add(Vec3A::splat(dt), self.linvel)
    }
}

pub struct Simulation {
    pub indexes: Arena<()>,
    pub handles: HashSet<Index>,
    pub positions: Vec<Vec3A>,
    pub linvels: Vec<Vec3A>,
    pub angvels: Vec<Vec3A>,
    pub masses: Vec<f32>,
    pub radii: Vec<f32>,

    pub octree: Octree,
    pub collisons: Vec<(Index, Index)>,

    pub total_mass: f32,
    pub center_of_mass: Vec3A,
    pub total_linear_momentum: Vec3A,
    pub total_angular_momentum: Vec3A,
    pub total_inertia: Mat3A,
    pub max_dist_sq: f32,
}

impl Simulation {
    pub fn new(grav_const: f32, theta: f32, epsilon: f32, max_cog_dist: f32) -> Self {
        Self {
            indexes: Arena::new(),
            handles: HashSet::new(),
            positions: Vec::new(),
            linvels: Vec::new(),
            angvels: Vec::new(),
            masses: Vec::new(),
            radii: Vec::new(),
            octree: Octree::new(grav_const, theta, epsilon),
            collisons: Vec::new(),

            total_mass: 0.0,
            center_of_mass: Vec3A::ZERO,
            total_linear_momentum: Vec3A::ZERO,
            total_angular_momentum: Vec3A::ZERO,
            total_inertia: Mat3A::ZERO,
            max_dist_sq: max_cog_dist * max_cog_dist,
        }
    }

    pub fn insert(&mut self, body: Body) -> Index {
        let len = self.positions.len();
        let handle = self.indexes.insert(());
        self.handles.insert(handle);
        let slot = handle.slot() as usize;
        if slot < len {
            self.positions[slot] = body.position;
            self.linvels[slot] = body.linvel;
            self.angvels[slot] = body.angvel;
            self.masses[slot] = body.mass;
            self.radii[slot] = body.radius;
        } else {
            if slot > len {
                self.positions.resize(slot, Vec3A::ZERO);
                self.linvels.resize(slot, Vec3A::ZERO);
                self.angvels.resize(slot, Vec3A::ZERO);
                self.masses.resize(slot, 0.0);
                self.radii.resize(slot, 0.0);
            }
            self.positions.push(body.position);
            self.linvels.push(body.linvel);
            self.angvels.push(body.angvel);
            self.masses.push(body.mass);
            self.radii.push(body.radius);
        }
        handle
    }

    fn remove(&mut self, handle: Index) -> Option<Body> {
        if self.indexes.remove(handle).is_some() {
            self.handles.remove(&handle);
            let slot = handle.slot() as usize;
            Some(Body {
                position: std::mem::replace(&mut self.positions[slot], Vec3A::ZERO),
                linvel: std::mem::replace(&mut self.linvels[slot], Vec3A::ZERO),
                angvel: std::mem::replace(&mut self.angvels[slot], Vec3A::ZERO),
                mass: std::mem::replace(&mut self.masses[slot], 0.0),
                radius: std::mem::replace(&mut self.radii[slot], 0.0),
            })
        } else {
            None
        }
    }

    fn combine(&mut self, handles: &[Index]) -> Index {
        const FRAC_4_3_PI: f32 = 4.0 * std::f32::consts::FRAC_PI_3;

        let (total_mass, mut center_of_mass) = handles.iter().fold(
            (0.0f32, Vec3A::ZERO),
            |(mut total_mass, mut center_of_mass), handle| {
                if self.indexes.contains(*handle) {
                    let slot = handle.slot() as usize;
                    let mass = self.masses[slot];
                    let center = self.positions[slot];
                    total_mass += mass;
                    center_of_mass += center * mass;
                }

                (total_mass, center_of_mass)
            },
        );
        center_of_mass /= total_mass;

        let (lin_moment, ang_moment, inertia) = handles.iter().fold(
            (Vec3A::ZERO, Vec3A::ZERO, Mat3A::ZERO),
            |(mut lin_moment, mut ang_moment, mut inertia_total), handle| {
                if self.indexes.contains(*handle) {
                    let slot = handle.slot() as usize;
                    let mass = self.masses[slot];
                    let center = self.positions[slot];
                    let linvel = self.linvels[slot];
                    let angvel = self.angvels[slot];
                    let radius = self.radii[slot];

                    let inertia = Mat3A::from_diagonal(Vec3::splat(mass * radius * radius * 0.4));
                    let dist = center - center_of_mass;
                    let dist_outer = Mat3A::from_cols(dist * dist.x, dist * dist.y, dist * dist.z);
                    let shift = Mat3A::IDENTITY * dist.length_squared() - dist_outer;

                    lin_moment += linvel * mass;
                    ang_moment += dist.cross(linvel * mass) + inertia * angvel;
                    inertia_total += inertia + shift * mass;
                }

                (lin_moment, ang_moment, inertia_total)
            },
        );

        let linvel = lin_moment / total_mass;
        let angvel = inertia.inverse() * ang_moment;

        let total_volume = handles
            .iter()
            .map(|handle| {
                if self.indexes.contains(*handle) {
                    let slot = handle.slot() as usize;
                    let radius = self.radii[slot];
                    radius * radius * radius * FRAC_4_3_PI
                } else {
                    0.0
                }
            })
            .sum::<f32>();

        let radius = (total_volume / FRAC_4_3_PI).cbrt();

        for handle in &handles[1..] {
            self.remove(*handle);
        }

        let slot = handles[0].slot() as usize;
        self.positions[slot] = center_of_mass;
        self.linvels[slot] = linvel;
        self.angvels[slot] = angvel;
        self.masses[slot] = total_mass;
        self.radii[slot] = radius;

        handles[0]
    }

    pub fn rebuild_octree(&mut self) {
        self.octree
            .build(&self.handles, &self.positions, &self.masses);
    }

    fn calc_cog_state(&mut self) {
        let (total_mass, mut center_of_mass) = self
            .masses
            .par_iter()
            .copied()
            .zip_eq(self.positions.par_iter().copied())
            .fold(
                || (0.0f32, Vec3A::ZERO),
                |(total_mass, center_of_mass), (mass, position)| {
                    (total_mass + mass, center_of_mass + position * mass)
                },
            )
            .reduce(
                || (0.0f32, Vec3A::ZERO),
                |(ma, pa), (mb, pb)| (ma + mb, pa + pb),
            );
        center_of_mass /= total_mass;

        let (lin_moment, ang_moment, inertia) = self
            .masses
            .par_iter()
            .copied()
            .zip_eq(self.radii.par_iter().copied())
            .zip_eq(
                self.linvels
                    .par_iter()
                    .copied()
                    .zip_eq(self.angvels.par_iter().copied())
                    .zip_eq(self.positions.par_iter().copied()),
            )
            .fold(
                || (Vec3A::ZERO, Vec3A::ZERO, Mat3A::ZERO),
                |(lin_moment, ang_moment, inertia_total),
                 ((mass, radius), ((linvel, angvel), position))| {
                    let inertia = Mat3A::from_diagonal(Vec3::splat(mass * radius * radius * 0.4));
                    let dist = position - center_of_mass;
                    let dist_outer = Mat3A::from_cols(dist * dist.x, dist * dist.y, dist * dist.z);
                    let shift = Mat3A::IDENTITY * dist.length_squared() - dist_outer;
                    (
                        lin_moment + linvel * mass,
                        ang_moment + dist.cross(linvel * mass) + inertia * angvel,
                        inertia_total + inertia + shift * mass,
                    )
                },
            )
            .reduce(
                || (Vec3A::ZERO, Vec3A::ZERO, Mat3A::ZERO),
                |(lma, ama, ia), (lmb, amb, ib)| (lma + lmb, ama + amb, ia + ib),
            );

        self.total_mass = total_mass;
        self.center_of_mass = center_of_mass;
        self.total_linear_momentum = lin_moment;
        self.total_angular_momentum = ang_moment;
        self.total_inertia = inertia;
    }

    fn leapfrog_integrate(&mut self, dt: f32) {
        const Y0: f32 = -1.7024144;
        const Y1: f32 = 1.3512071;
        const Y2: f32 = 0.67560357;
        const Y3: f32 = -0.1756036;

        if dt == 0.0 {
            return;
        }

        let y0dt = Y0 * dt;
        let y1dt = Y1 * dt;
        let y2dt = Y2 * dt;
        let y3dt = Y3 * dt;

        self.positions
            .par_iter_mut()
            .zip_eq(self.linvels.par_iter())
            .for_each(|(position, velocity)| {
                // Step 1a: r1 = r0 + c2*v0*dt
                *position = (*velocity).mul_add(Vec3A::splat(y2dt), *position);
            });

        self.rebuild_octree();
        self.positions
            .par_iter_mut()
            .zip_eq(self.linvels.par_iter_mut())
            .for_each(|(position, velocity)| {
                // Step 1b: v1 = v0 + c1*a(r1)*dt
                let accel = self.octree.accel_at(*position);
                *velocity = accel.mul_add(Vec3A::splat(y1dt), *velocity);

                // Step 2a: r2 = r1 + c3*v1*dt
                *position = (*velocity).mul_add(Vec3A::splat(y3dt), *position);
            });

        self.rebuild_octree();
        self.positions
            .par_iter_mut()
            .zip_eq(self.linvels.par_iter_mut())
            .for_each(|(position, velocity)| {
                // Step 2b: v2 = v1 + c0*a(r2)*dt
                let accel = self.octree.accel_at(*position);
                *velocity = accel.mul_add(Vec3A::splat(y0dt), *velocity);

                // Step 3a: r3 = r2 + c3*v2*dt
                *position = (*velocity).mul_add(Vec3A::splat(y3dt), *position);
            });

        self.rebuild_octree();
        self.positions
            .par_iter_mut()
            .zip_eq(self.linvels.par_iter_mut())
            .for_each(|(position, velocity)| {
                // Step 3b: v3 = v2 + c1*a(r3)*dt
                let accel = self.octree.accel_at(*position);
                *velocity = accel.mul_add(Vec3A::splat(y1dt), *velocity);

                // Step 4a: r4 = r3 + c2*v3*dt
                *position = (*velocity).mul_add(Vec3A::splat(y2dt), *position);
            });
    }

    fn resolve_collisions(&mut self) {
        self.rebuild_octree();
        let collisions = self
            .handles
            .par_iter()
            .copied()
            .filter_map(|handle| {
                let this_slot = handle.slot() as usize;
                if let Some((nearest, dist)) = self
                    .octree
                    .closest_at(self.positions[this_slot], Some(handle))
                {
                    let other_slot = nearest.slot() as usize;
                    if dist <= self.radii[this_slot].max(self.radii[other_slot])
                        && self.masses[other_slot] >= self.masses[this_slot]
                    {
                        Some((nearest, handle))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut remap = HashMap::<Index, Index>::new();

        for (a, b) in &collisions {
            remap.insert(*a, *a);
            remap.insert(*b, *b);
        }

        for (a, b) in collisions {
            let a = remap[&a];
            let b = remap[&b];
            if a != b {
                let new = self.combine(&[a, b]);
                remap.insert(a, new);
                remap.insert(b, new);
            }
        }

        // drop(sender);
        // self.collisons.clear();

        // self.collisons.extend(reciever.iter());

        // for col_index in 0..self.collisons.len() {
        //     let (j, i) = self.collisons[col_index];
        //     self.combine(j, i);
        //     let new = i;
        //     for (j, i) in self.collisons[col_index + 1..].iter_mut() {
        //         if *j == self.bodies.len() {
        //             *j = new;
        //         }
        //         if *i == self.bodies.len() {
        //             *i = new;
        //         }
        //     }
        // }
    }

    fn resolve_out_of_bounds(&mut self) {
        self.rebuild_octree();
        let handles_to_remove = self
            .handles
            .par_iter()
            .copied()
            .filter(|handle| {
                let slot = handle.slot() as usize;
                let dist_sq = self.center_of_mass.distance_squared(self.positions[slot]);
                if dist_sq > self.max_dist_sq * 16.0 {
                    true
                } else if dist_sq > self.max_dist_sq {
                    let dist = dist_sq.sqrt();
                    let vel_sq = self.linvels[slot].length_squared();
                    let accel = self.octree.accel_at(self.positions[slot]).length();

                    vel_sq < 2.0 * accel * dist
                } else {
                    false
                }
            })
            .collect::<Vec<_>>();

        handles_to_remove.into_iter().for_each(|handle| {
            self.remove(handle);
        });
    }

    pub fn total_angular_momentum(&self) -> f32 {
        self.total_angular_momentum.length()
    }

    pub fn step(&mut self, dt: f32) {
        // while dt > max_step {
        //     self.leapfrog_integrate(max_step);
        //     dt -= max_step;
        // }

        self.leapfrog_integrate(dt);

        self.resolve_collisions();

        self.calc_cog_state();

        self.resolve_out_of_bounds();

        self.calc_cog_state();
    }
}

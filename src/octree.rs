use std::{
    collections::HashSet,
    num::NonZero,
    ops::Range,
    sync::{Arc, Weak, atomic::AtomicUsize},
};

use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use thunderdome::{Arena, Index};

use crate::{math::*, sim::Body};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cube(Vec4);

impl Default for Cube {
    fn default() -> Self {
        Self::ZEROED
    }
}

impl Cube {
    pub const ZEROED: Self = Self(Vec4::ZERO);

    pub fn new_containing(points: impl Iterator<Item = Vec3A>) -> Self {
        let mut min = Vec3A::MAX;
        let mut max = Vec3A::MIN;

        for point in points {
            min = min.min(point);
            max = max.max(point);
        }

        Self::new_min_max(min, max)
    }

    pub fn new_min_max(min: Vec3A, max: Vec3A) -> Self {
        let center = (min + max) * 0.5;
        let size = (max - min).max_element();

        Self(center.extend(size))
    }

    pub fn center(&self) -> Vec3A {
        Vec3A::from_vec4(self.0)
    }

    pub fn size(&self) -> f32 {
        self.0.w
    }

    pub fn into_octant(&self, octant: BVec3A) -> Self {
        const HALF: Vec3A = Vec3A::splat(0.5);
        const NEG_HALF: Vec3A = Vec3A::splat(-0.5);
        let size = self.size() * 0.5;
        let center = self.center() + Vec3A::select(octant, HALF, NEG_HALF) * size;
        Self(center.extend(size))
    }

    pub fn subdivide(&self) -> [Cube; 8] {
        [
            self.into_octant(BVec3A::new(false, false, false)),
            self.into_octant(BVec3A::new(true, false, false)),
            self.into_octant(BVec3A::new(false, true, false)),
            self.into_octant(BVec3A::new(true, true, false)),
            self.into_octant(BVec3A::new(false, false, true)),
            self.into_octant(BVec3A::new(true, false, true)),
            self.into_octant(BVec3A::new(false, true, true)),
            self.into_octant(BVec3A::new(true, true, true)),
        ]
    }

    pub fn contains(&self, point: Vec3A) -> bool {
        let center = self.center();
        let hi_bound = center + self.size() * 0.5;
        let lo_bound = center - self.size() * 0.5;

        (point.cmple(hi_bound) & point.cmpge(lo_bound)).all()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Node {
    parent: usize,
    children: Option<NonZero<usize>>,
    next: Option<NonZero<usize>>,
    bounds: Cube,
    center_of_mass: Vec3A,
    total_mass: f32,
    body: Option<Index>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Octree {
    pub g: f32,
    pub t_sq: f32,
    pub e_sq: f32,
    nodes: Vec<Node>,
    bodies: Vec<Index>,
}

impl Octree {
    pub fn new(grav_const: f32, theta: f32, epsilon: f32) -> Self {
        Self {
            g: grav_const,
            t_sq: theta * theta,
            e_sq: epsilon * epsilon,
            nodes: Vec::new(),
            bodies: Vec::new(),
        }
    }

    /// O(n*log(n)) because of sort
    fn subdivide_if_needed<'a>(
        index: usize,
        count: &AtomicUsize,
        node: &mut Node,
        handles: &'a mut [Index],
        positions: &[Vec3A],
    ) -> Option<[(usize, Node, &'a mut [Index]); 8]> {
        let center = node.bounds.center();

        if handles.len() <= 1 {
            if let Some(&handle) = handles.first() {
                node.body = Some(handle);
            }
            return None;
        }

        handles.sort_unstable_by_key(|handle| {
            positions[handle.slot() as usize].cmpgt(center).bitmask()
        });

        let predicate = |handle: &Index| positions[handle.slot() as usize].z <= center.z;
        let (lz, gz) = handles.split_at_mut(handles.partition_point(&predicate));

        let predicate = |handle: &Index| positions[handle.slot() as usize].y <= center.y;
        let (lz_ly, lz_gy) = lz.split_at_mut(lz.partition_point(&predicate));
        let (gz_ly, gz_gy) = gz.split_at_mut(gz.partition_point(&predicate));

        let predicate = |handle: &Index| positions[handle.slot() as usize].x <= center.x;
        let (lz_ly_lx, lz_ly_gx) = lz_ly.split_at_mut(lz_ly.partition_point(&predicate));
        let (lz_gy_lx, lz_gy_gx) = lz_gy.split_at_mut(lz_gy.partition_point(&predicate));
        let (gz_ly_lx, gz_ly_gx) = gz_ly.split_at_mut(gz_ly.partition_point(&predicate));
        let (gz_gy_lx, gz_gy_gx) = gz_gy.split_at_mut(gz_gy.partition_point(&predicate));

        let base = count.fetch_add(8, std::sync::atomic::Ordering::Relaxed);

        let mut ranges = [
            lz_ly_lx, lz_ly_gx, lz_gy_lx, lz_gy_gx, gz_ly_lx, gz_ly_gx, gz_gy_lx, gz_gy_gx,
        ];

        let octants = node.bounds.subdivide();

        node.children = NonZero::new(base);

        let nexts = [
            NonZero::new(base + 1),
            NonZero::new(base + 2),
            NonZero::new(base + 3),
            NonZero::new(base + 4),
            NonZero::new(base + 5),
            NonZero::new(base + 6),
            NonZero::new(base + 7),
            node.next,
        ];

        Some(std::array::from_fn(move |i| {
            (
                base + i,
                Node {
                    parent: index,
                    children: None,
                    next: nexts[i],
                    bounds: octants[i],
                    center_of_mass: octants[i].center(),
                    total_mass: 0.0,
                    body: None,
                },
                std::mem::replace(&mut ranges[i], &mut []),
            )
        }))
    }

    /// O(n)
    fn propogate(&mut self, positions: &[Vec3A], masses: &[f32]) {
        for node in (0..self.nodes.len()).rev() {
            let mut center_of_mass = Vec3A::ZERO;
            let mut total_mass = 0.0f32;
            if let Some(children) = self.nodes[node].children {
                let children = children.get();
                for child in children..children + 8 {
                    let mass = self.nodes[child].total_mass;
                    center_of_mass += self.nodes[child].center_of_mass * mass;
                    total_mass += mass;
                }
            } else {
                if let Some(handle) = self.nodes[node].body {
                    let mass = masses[handle.slot() as usize];
                    center_of_mass = positions[handle.slot() as usize] * mass;
                    total_mass = mass;
                }
            }

            self.nodes[node].center_of_mass = if total_mass > 0.0 {
                center_of_mass / total_mass
            } else {
                self.nodes[node].bounds.center()
            };
            self.nodes[node].total_mass = total_mass;
        }
    }

    pub fn is_built(&self) -> bool {
        !self.nodes.is_empty()
    }

    pub fn invalidate(&mut self) {
        self.nodes.clear();
    }

    pub fn build(&mut self, indexes: &HashSet<Index>, positions: &[Vec3A], masses: &[f32]) {
        self.invalidate();
        let mut handles = std::mem::take(&mut self.bodies);
        handles.clear();
        handles.par_extend(indexes);

        let (min_bound, max_bound) = indexes
            .into_par_iter()
            .fold(
                || (Vec3A::MAX, Vec3A::MIN),
                |(min, max), handle| {
                    let position = positions[handle.slot() as usize];
                    (min.min(position), max.max(position))
                },
            )
            .reduce(
                || (Vec3A::MAX, Vec3A::MIN),
                |(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)),
            );
        let bounds = Cube::new_min_max(min_bound, max_bound);

        let count = Arc::new(AtomicUsize::new(1));
        let processing = Arc::new(AtomicUsize::new(1));
        let (job_sender, job_reciever) = crossbeam::channel::unbounded();

        job_sender
            .send((
                0usize,
                Node {
                    parent: 0,
                    children: None,
                    next: None,
                    bounds,
                    center_of_mass: bounds.center(),
                    total_mass: 0.0,
                    body: None,
                },
                handles.as_mut_slice(),
            ))
            .unwrap();

        let nodes = rayon::broadcast(|_| {
            let mut nodes = Vec::new();
            let processing = processing.clone();
            let count = count.clone();
            while processing.load(std::sync::atomic::Ordering::SeqCst) > 0 {
                while let Ok((index, mut node, handles)) = job_reciever.try_recv() {
                    if let Some(results) =
                        Self::subdivide_if_needed(index, &count, &mut node, handles, positions)
                    {
                        for results in results {
                            job_sender.send(results).unwrap();
                        }
                        processing.fetch_add(8, std::sync::atomic::Ordering::SeqCst);
                    }
                    nodes.push((index, node));
                    processing.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
            nodes
        });

        assert!(job_reciever.len() == 0, "not all jobs processed");
        assert!(
            processing.load(std::sync::atomic::Ordering::Relaxed) == 0,
            "processing should be 0"
        );

        self.nodes.resize(
            count.load(std::sync::atomic::Ordering::Relaxed),
            Node::default(),
        );
        nodes.into_iter().flatten().for_each(|(index, node)| {
            self.nodes[index] = node;
        });

        self.propogate(positions, masses);
    }

    pub fn accel_at(&self, pos: Vec3A) -> Vec3A {
        assert!(self.is_built(), "Octree was queried before it was built!");
        let mut total_accel = Vec3A::ZERO;

        let mut index = 0usize;
        loop {
            let node = &self.nodes[index];

            let dist = node.center_of_mass - pos;
            let dist_sq = dist.length_squared();
            let node_mass = node.total_mass;

            if node.bounds.size() * node.bounds.size() < dist_sq * self.t_sq {
                if dist_sq > 0.0 {
                    let denom = (dist_sq + self.e_sq) * dist_sq.sqrt();
                    let accel = dist * (self.g * node_mass / denom);
                    // println!(
                    //     "Applied indirect accel {} to mass from {} to {}",
                    //     accel,
                    //     dist + pos,
                    //     pos
                    // );

                    total_accel += accel;
                }

                if let Some(i) = node.next {
                    index = i.get();
                } else {
                    break;
                }
            } else if node.children.is_none() {
                if let Some(handle) = node.body {
                    // let body = &bodies[handle];
                    // let dist = body.position() - pos;
                    // let dist_sq = dist.length_squared();
                    // let node_mass = node.total_mass;

                    if dist_sq > 1.0e-5 && !node.bounds.contains(pos) {
                        let denom = (dist_sq + self.e_sq) * dist_sq.sqrt();
                        let accel = dist * (self.g * node_mass / denom);
                        // println!(
                        //     "Applied direct accel {} to mass  from {} to {} ({})",
                        //     accel,
                        //     dist + pos,
                        //     pos,
                        //     node_mass
                        // );
                        total_accel += accel;
                    }
                }

                if let Some(i) = node.next {
                    index = i.get();
                } else {
                    break;
                }
            } else {
                if let Some(i) = node.children {
                    assert!(self.nodes[i.get()].parent == index);
                    index = i.get();
                } else {
                    break;
                }
            }
        }

        total_accel
    }

    pub fn force_at(&self, pos: Vec3A, mass: f32) -> Vec3A {
        self.accel_at(pos) * mass
    }

    pub fn closest_at(&self, pos: Vec3A, exclude: Option<Index>) -> Option<(Index, f32)> {
        assert!(self.is_built(), "Octree was queried before it was built!");
        let mut closest_dist = f32::INFINITY;
        let mut closest = None;

        let mut index = 0usize;
        loop {
            let node = &self.nodes[index];

            if let Some(children) = node.children {
                index = children.get();
            } else {
                if let Some(handle) = node.body {
                    let dist = pos.distance_squared(node.center_of_mass);
                    if dist < closest_dist && Some(handle) != exclude {
                        closest_dist = dist;
                        closest.replace(handle);
                    }
                }

                if let Some(next) = node.next {
                    index = next.get()
                } else {
                    break;
                }
            }
        }

        closest.map(|handle| (handle, closest_dist.sqrt()))
    }
}

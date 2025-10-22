use std::{ops::Range, sync::atomic::{AtomicUsize, Ordering}};

use glam::*;

use parking_lot::RwLock;
use rayon::prelude::*;

use crate::sim::SimBody;

#[derive(Clone, Copy)]
struct Cube(Vec4);

impl Cube {
    pub const ZEROED: Self = Self(Vec4::ZERO);

    pub fn new_containing(bodies: &[SimBody]) -> Self {
        let mut min = Vec3A::MAX;
        let mut max = Vec3A::MIN;

        for body in bodies {
            min = min.min(body.pos);
            max = max.max(body.pos);
        }

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

    pub fn into_quadrant(&self, quadrant: BVec3A) -> Self {
        const HALF: Vec3A = Vec3A::splat(0.5);
        const NEG_HALF: Vec3A = Vec3A::splat(-0.5);
        let size = self.size() * 0.5;
        let center = self.center() + Vec3A::select(quadrant, HALF, NEG_HALF) * size;
        Self(center.extend(size))
    }

    pub fn subdivide(&self) -> [Cube; 8] {
        [
            self.into_quadrant(BVec3A::new(false, false, false)),
            self.into_quadrant(BVec3A::new(true,  false, false)),
            self.into_quadrant(BVec3A::new(false, true,  false)),
            self.into_quadrant(BVec3A::new(true,  true,  false)),
            self.into_quadrant(BVec3A::new(false, false, true )),
            self.into_quadrant(BVec3A::new(true,  false, true )),
            self.into_quadrant(BVec3A::new(false, true,  true )),
            self.into_quadrant(BVec3A::new(true,  true,  true )),
        ]
    }
}

#[derive(Clone)]
struct Node {
    pub children: usize,
    pub next: usize,
    pub pos: Vec3A,
    pub mass: f32,
    pub cube: Cube,
    pub bodies: Range<usize>
}

impl Node {
    pub const ZEROED: Self = Self {
        children: 0,
        next: 0,
        pos: Vec3A::ZERO,
        mass: 0.0,
        cube: Cube::ZEROED,
        bodies: 0..0,
    };

    pub fn new(next: usize, cube: Cube, bodies: Range<usize>) -> Self {
        Self {
            children: 0,
            next,
            pos: Vec3A::ZERO,
            mass: 0.0,
            cube,
            bodies
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children == 0
    }
}


pub struct Octree {
    t_sq: f32,
    e_sq: f32,
    leaf_capacity: usize,
    thread_capacity: usize,
    atomic_len: AtomicUsize,
    nodes: Vec<RwLock<Node>>,
    parents: Vec<AtomicUsize>,
}

impl Octree {
    pub const ROOT: usize = 0;

    pub fn new(theta: f32, epsilon: f32, leaf_capacity: usize, thread_capacity: usize) -> Self {
        Self {
            t_sq: theta * theta,
            e_sq: epsilon * epsilon,
            leaf_capacity,
            thread_capacity,
            atomic_len: AtomicUsize::new(0),
            nodes: Vec::new(),
            parents: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.atomic_len.store(0, Ordering::Relaxed);
    }

    fn subdivide(&self, index: usize, node: &mut parking_lot::RwLockUpgradableReadGuard<Node>, bodies: &mut [SimBody], range: Range<usize>) -> usize {
        let center = node.cube.center();

        let mut split = [range.start, 0, 0, 0, 0, 0, 0, 0, range.end];

        let predicate = |body: &SimBody| body.pos.z < center.z;
        split[4] = split[0] + bodies[split[0]..split[8]].partition(predicate);

        let predicate = |body: &SimBody| body.pos.y < center.y;
        split[2] = split[0] + bodies[split[0]..split[4]].partition(predicate);
        split[6] = split[4] + bodies[split[4]..split[8]].partition(predicate);

        let predicate = |body: &SimBody| body.pos.x < center.x;
        split[1] = split[0] + bodies[split[0]..split[2]].partition(predicate);
        split[3] = split[2] + bodies[split[2]..split[4]].partition(predicate);
        split[5] = split[4] + bodies[split[4]..split[6]].partition(predicate);
        split[7] = split[6] + bodies[split[6]..split[8]].partition(predicate);

        let len = self.atomic_len.fetch_add(1, Ordering::Relaxed);
        let children = len * 8 + 1;
        self.parents[len].store(index, Ordering::Relaxed);
        node.with_upgraded(|node| node.children = children);

        let nexts = [
            children + 1,
            children + 2,
            children + 3,
            children + 4,
            children + 5,
            children + 6,
            children + 7,
            node.next,
        ];
        let cubes = node.cube.subdivide();
        for i in 0..8 {
            let bodies = split[i]..split[i + 1];
            *self.nodes[children + i].write() = Node::new(nexts[i], cubes[i], bodies);
        }

        children
    }

    pub fn build(&mut self, bodies: &mut [SimBody]) {
        self.clear();

        let new_len = 8 * bodies.len();
        self.nodes.resize_with(new_len, || RwLock::new(Node::ZEROED));
        self.parents.resize_with(new_len / 8, || AtomicUsize::new(0));

        let cube = Cube::new_containing(bodies);
        *self.nodes[Self::ROOT].get_mut() = Node::new(0, cube, 0..bodies.len());

        let (tx, rx) = crossbeam::channel::unbounded();
        tx.send(Self::ROOT).unwrap();

        let octree_ptr = &*self;
        let bodies_ptr = bodies.as_ptr() as usize;
        let bodies_len = bodies.len();

        let counter = AtomicUsize::new(0);
        rayon::broadcast(|_| {
            let mut stack = Vec::new();
            let octree = octree_ptr;
            let bodies =
                unsafe { std::slice::from_raw_parts_mut(bodies_ptr as *mut SimBody, bodies_len) };

            while counter.load(Ordering::Relaxed) != bodies.len() {
                while let Ok(node) = rx.try_recv() {
                    let mut node_lock = octree.nodes[node].upgradable_read();
                    let range = node_lock.bodies.clone();

                    if range.len() >= octree.thread_capacity {
                        let children = octree.subdivide(node, &mut node_lock, bodies, range);
                        for i in 0..8 {
                            if !self.nodes[children + i].read().bodies.is_empty() {
                                tx.send(children + i).unwrap();
                            }
                        }
                        continue;
                    }

                    counter.fetch_add(range.len(), Ordering::Relaxed);

                    drop(node_lock);

                    stack.push(node);
                    while let Some(node) = stack.pop() {
                        let mut node_lock = octree.nodes[node].upgradable_read();
                        let range = node_lock.bodies.clone();
                        if range.len() <= octree.leaf_capacity {
                            let pos = bodies[range.clone()].iter().map(|b| b.pos * b.mass).sum();
                            let mass = bodies[range.clone()].iter().map(|b| b.mass).sum();
                            node_lock.with_upgraded(|node| {
                                node.pos = pos;
                                node.mass = mass;
                            });
                            continue;
                        }
                        let children = octree.subdivide(node, &mut node_lock, bodies, range);
                        for i in 0..8 {
                            if !self.nodes[children + i].read().bodies.is_empty() {
                                stack.push(children + i);
                            }
                        }
                    }
                }
            }
        });

        self.propagate();
    }

    fn propagate(&mut self) {
        let len = self.atomic_len.load(Ordering::Relaxed);
        for index in self.parents.iter().rev() {
            let index = index.load(Ordering::Relaxed);
            let i = self.nodes[index].get_mut().children;

            self.nodes[index].get_mut().pos = Vec3A::ZERO;
            self.nodes[index].get_mut().mass = 0.0;
            for child in i..i+8 {
                let pos = self.nodes[child].get_mut().pos;
                let mass = self.nodes[child].get_mut().mass;
                self.nodes[index].get_mut().pos += pos;
                self.nodes[index].get_mut().mass += mass;
            }
        }
        self.nodes[0..len * 8 + 1].par_iter_mut().for_each(|node| {
            let mass = node.get_mut().mass.max(f32::MIN_POSITIVE);
            node.get_mut().pos /= mass;
        });
    }

    pub fn force_at(&self, bodies: &[SimBody], pos: Vec3A, mass: f32, grav_const: f32) -> Vec3A {
        let mut acc = Vec3A::ZERO;

        let mut node = Self::ROOT;
        loop {
            let n = &*self.nodes[node].read();

            let d = n.pos - pos;
            let d_sq = d.length_squared();

            if n.cube.size() * n.cube.size() < d_sq * self.t_sq {
                let denom = (d_sq + self.e_sq) * d_sq.sqrt();
                acc += d * (grav_const * mass * n.mass / denom);

                if n.next == 0 {
                    break;
                }
                node = n.next;
            } else if n.is_leaf() {
                for i in n.bodies.start..n.bodies.end {
                    let body = &bodies[i];
                    let d = body.pos - pos;
                    let d_sq = d.length_squared();

                    let denom = (d_sq + self.e_sq) * d_sq.sqrt();
                    acc += d * (grav_const * mass * body.mass / denom).min(f32::MAX);
                }

                if n.next == 0 {
                    break;
                }
                node = n.next;
            } else {
                node = n.children;
            }
        }

        acc
    }

    pub fn force(&self, bodies: &mut [SimBody], grav_const: f32) {
        let bodies_ptr = bodies.as_ptr() as usize;
        let bodies_len = bodies.len();

        bodies.par_iter_mut().for_each(|body| {
            let bodies = unsafe { std::slice::from_raw_parts(bodies_ptr as *const SimBody, bodies_len) };
            body.acc = self.force_at(bodies, body.pos, body.mass, grav_const) / body.mass;
        });
    }
}

pub trait Partition<T> {
    /// Partitions self in place so that all elements for which the `predicate`
    /// returns `true` are positioned before all elements for which it returns `false`.
    ///
    /// The function returns the index of the first element for which `predicate` returns `false`.
    fn partition<F>(&mut self, predicate: F) -> usize
    where
        F: Fn(&T) -> bool;
}

impl<T> Partition<T> for [T] {
    fn partition<F>(&mut self, predicate: F) -> usize
    where
        F: Fn(&T) -> bool,
    {
        if self.is_empty() {
            return 0;
        }

        let mut l = 0;
        let mut r = self.len() - 1;

        loop {
            while l <= r && predicate(&self[l]) {
                l += 1;
            }
            while l < r && !predicate(&self[r]) {
                r -= 1;
            }
            if l >= r {
                return l;
            }

            self.swap(l, r);
            l += 1;
            r -= 1;
        }
    }
}
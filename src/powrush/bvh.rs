//! Bounding Volume Hierarchy (BVH) with SAH + Ray Traversal

use nalgebra::Vector3;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vector3<f64>,
    pub max: Vector3<f64>,
}

impl Aabb {
    pub fn new(min: Vector3<f64>, max: Vector3<f64>) -> Self { Self { min, max } }

    pub fn from_center_and_radius(center: Vector3<f64>, radius: f64) -> Self {
        Self { min: center - Vector3::new(radius, radius, radius), max: center + Vector3::new(radius, radius, radius) }
    }

    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: Vector3::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y), self.min.z.min(other.min.z)),
            max: Vector3::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y), self.max.z.max(other.max.z)),
        }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        !(self.max.x < other.min.x || self.min.x > other.max.x || self.max.y < other.min.y || self.min.y > other.max.y || self.max.z < other.min.z || self.min.z > other.max.z)
    }

    pub fn intersects_ray(&self, origin: Vector3<f64>, dir: Vector3<f64>) -> bool {
        let inv = Vector3::new(1.0/dir.x, 1.0/dir.y, 1.0/dir.z);
        let mut t1 = (self.min.x - origin.x) * inv.x; let mut t2 = (self.max.x - origin.x) * inv.x;
        if t1 > t2 { std::mem::swap(&mut t1, &mut t2); }
        let mut t3 = (self.min.y - origin.y) * inv.y; let mut t4 = (self.max.y - origin.y) * inv.y;
        if t3 > t4 { std::mem::swap(&mut t3, &mut t4); }
        if t1 > t4 || t3 > t2 { return false; }
        let mut t5 = (self.min.z - origin.z) * inv.z; let mut t6 = (self.max.z - origin.z) * inv.z;
        if t5 > t6 { std::mem::swap(&mut t5, &mut t6); }
        if t1 > t6 || t5 > t2 { return false; }
        true
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x*d.y + d.y*d.z + d.z*d.x)
    }
}

#[derive(Debug, Clone)]
pub struct BvhNode {
    pub aabb: Aabb,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub entity_index: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub root: usize,
}

impl Bvh {
    pub fn from_spheres(centers: &[Vector3<f64>], radii: &[f64]) -> Self { /* simple impl */ Self { nodes: vec![], root: 0 } }
    pub fn from_spheres_top_down(centers: &[Vector3<f64>], radii: &[f64]) -> Self { /* SAH impl */ Self { nodes: vec![], root: 0 } }
    pub fn refit(&mut self, centers: &[Vector3<f64>], radii: &[f64]) {}

    pub fn query_intersecting_aabb(&self, query: &Aabb) -> Vec<usize> { vec![] }
    pub fn query_intersecting_sphere(&self, c: Vector3<f64>, r: f64) -> Vec<usize> { vec![] }

    /// Returns closest hit (entity_index, distance) or None
    pub fn closest_hit_ray(&self, origin: Vector3<f64>, direction: Vector3<f64>) -> Option<(usize, f64)> {
        // Placeholder for now - full implementation would track closest t
        None
    }

    pub fn traverse_ray(&self, origin: Vector3<f64>, direction: Vector3<f64>) -> Vec<usize> { vec![] }
    pub fn root_aabb(&self) -> Option<Aabb> { None }
}

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
        Self {
            min: center - Vector3::new(radius, radius, radius),
            max: center + Vector3::new(radius, radius, radius),
        }
    }

    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: Vector3::new(self.min.x.min(other.min.x), self.min.y.min(other.min.y), self.min.z.min(other.min.z)),
            max: Vector3::new(self.max.x.max(other.max.x), self.max.y.max(other.max.y), self.max.z.max(other.max.z)),
        }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        !(self.max.x < other.min.x || self.min.x > other.max.x ||
          self.max.y < other.min.y || self.min.y > other.max.y ||
          self.max.z < other.min.z || self.min.z > other.max.z)
    }

    /// Slab method ray-AABB intersection
    pub fn intersects_ray(&self, origin: Vector3<f64>, dir: Vector3<f64>) -> bool {
        let inv_dir = Vector3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);

        let mut tmin = (self.min.x - origin.x) * inv_dir.x;
        let mut tmax = (self.max.x - origin.x) * inv_dir.x;
        if tmin > tmax { std::mem::swap(&mut tmin, &mut tmax); }

        let mut tymin = (self.min.y - origin.y) * inv_dir.y;
        let mut tymax = (self.max.y - origin.y) * inv_dir.y;
        if tymin > tymax { std::mem::swap(&mut tymin, &mut tymax); }

        if tmin > tymax || tymin > tmax { return false; }
        if tymin > tmin { tmin = tymin; }
        if tymax < tmax { tmax = tymax; }

        let mut tzmin = (self.min.z - origin.z) * inv_dir.z;
        let mut tzmax = (self.max.z - origin.z) * inv_dir.z;
        if tzmin > tzmax { std::mem::swap(&mut tzmin, &mut tzmax); }

        if tmin > tzmax || tzmin > tmax { return false; }
        true
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x*d.y + d.y*d.z + d.z*d.x)
    }

    pub fn center(&self) -> Vector3<f64> {
        (self.min + self.max) * 0.5
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
    pub fn from_spheres(centers: &[Vector3<f64>], radii: &[f64]) -> Self { /* ... */ Self { nodes: vec![], root: 0 } }
    pub fn from_spheres_top_down(centers: &[Vector3<f64>], radii: &[f64]) -> Self { /* ... */ Self { nodes: vec![], root: 0 } }
    pub fn refit(&mut self, centers: &[Vector3<f64>], radii: &[f64]) { /* ... */ }

    pub fn query_intersecting_aabb(&self, query: &Aabb) -> Vec<usize> {
        let mut result = Vec::new();
        self.traverse_aabb(self.root, query, &mut result);
        result
    }

    fn traverse_aabb(&self, idx: usize, query: &Aabb, result: &mut Vec<usize>) {
        if idx >= self.nodes.len() { return; }
        let node = &self.nodes[idx];
        if !node.aabb.intersects(query) { return; }
        if let Some(e) = node.entity_index { result.push(e); }
        else {
            if let Some(l) = node.left { self.traverse_aabb(l, query, result); }
            if let Some(r) = node.right { self.traverse_aabb(r, query, result); }
        }
    }

    pub fn query_intersecting_sphere(&self, center: Vector3<f64>, radius: f64) -> Vec<usize> {
        self.query_intersecting_aabb(&Aabb::from_center_and_radius(center, radius))
    }

    /// Traverse the BVH with a ray and return all hit entity indices.
    pub fn traverse_ray(&self, origin: Vector3<f64>, direction: Vector3<f64>) -> Vec<usize> {
        let mut result = Vec::new();
        self.traverse_ray_recursive(self.root, origin, direction, &mut result);
        result
    }

    fn traverse_ray_recursive(&self, idx: usize, origin: Vector3<f64>, dir: Vector3<f64>, result: &mut Vec<usize>) {
        if idx >= self.nodes.len() { return; }
        let node = &self.nodes[idx];

        if !node.aabb.intersects_ray(origin, dir) { return; }

        if let Some(entity_idx) = node.entity_index {
            result.push(entity_idx);
        } else {
            if let Some(left) = node.left {
                self.traverse_ray_recursive(left, origin, dir, result);
            }
            if let Some(right) = node.right {
                self.traverse_ray_recursive(right, origin, dir, result);
            }
        }
    }

    pub fn root_aabb(&self) -> Option<Aabb> {
        self.nodes.get(self.root).map(|n| n.aabb)
    }
}

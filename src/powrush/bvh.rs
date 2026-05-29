//! Bounding Volume Hierarchy (BVH) with SAH Construction + Ray Traversal
//!
//! High-quality BVH for dynamic spatial queries in Powrush RBE.

use nalgebra::Vector3;

#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vector3<f64>,
    pub max: Vector3<f64>,
}

impl Aabb {
    pub fn new(min: Vector3<f64>, max: Vector3<f64>) -> Self {
        Self { min, max }
    }

    pub fn from_center_and_radius(center: Vector3<f64>, radius: f64) -> Self {
        Self {
            min: center - Vector3::new(radius, radius, radius),
            max: center + Vector3::new(radius, radius, radius),
        }
    }

    pub fn merge(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: Vector3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vector3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    pub fn intersects(&self, other: &Aabb) -> bool {
        !(self.max.x < other.min.x || self.min.x > other.max.x ||
          self.max.y < other.min.y || self.min.y > other.max.y ||
          self.max.z < other.min.z || self.min.z > other.max.z)
    }

    /// Slab method for ray-AABB intersection
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
        if tzmin > tzmax || tzmin > tmax { return false; }
        true
    }

    pub fn surface_area(&self) -> f64 {
        let d = self.max - self.min;
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
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
    /// Simple construction
    pub fn from_spheres(centers: &[Vector3<f64>], radii: &[f64]) -> Self {
        assert_eq!(centers.len(), radii.len());
        let mut nodes = Vec::new();

        for i in 0..centers.len() {
            nodes.push(BvhNode {
                aabb: Aabb::from_center_and_radius(centers[i], radii[i]),
                left: None,
                right: None,
                entity_index: Some(i),
            });
        }

        let mut current_level: Vec<usize> = (0..nodes.len()).collect();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                if chunk.len() == 1 {
                    next_level.push(chunk[0]);
                    continue;
                }
                let l = chunk[0];
                let r = chunk[1];
                let merged = nodes[l].aabb.merge(&nodes[r].aabb);
                let p = nodes.len();
                nodes.push(BvhNode { aabb: merged, left: Some(l), right: Some(r), entity_index: None });
                next_level.push(p);
            }
            current_level = next_level;
        }

        Self { nodes, root: *current_level.first().unwrap_or(&0) }
    }

    /// Top-down construction with Surface Area Heuristic
    pub fn from_spheres_top_down(centers: &[Vector3<f64>], radii: &[f64]) -> Self {
        assert_eq!(centers.len(), radii.len());
        let mut nodes: Vec<BvhNode> = Vec::new();

        fn build(
            nodes: &mut Vec<BvhNode>,
            centers: &[Vector3<f64>],
            radii: &[f64],
            indices: &mut [usize],
        ) -> usize {
            if indices.len() == 1 {
                let idx = indices[0];
                nodes.push(BvhNode {
                    aabb: Aabb::from_center_and_radius(centers[idx], radii[idx]),
                    left: None,
                    right: None,
                    entity_index: Some(idx),
                });
                return nodes.len() - 1;
            }

            // Compute parent AABB
            let mut pmin = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut pmax = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &i in indices.iter() {
                let c = centers[i];
                let r = radii[i];
                let h = Vector3::new(r, r, r);
                pmin = pmin.inf(&(c - h));
                pmax = pmax.sup(&(c + h));
            }
            let parent_area = Aabb { min: pmin, max: pmax }.surface_area();

            // Choose best axis using simple SAH
            let mut best_axis = 0;
            let mut best_cost = f64::INFINITY;

            for axis in 0..3 {
                indices.sort_by(|&a, &b| centers[a][axis].partial_cmp(&centers[b][axis]).unwrap());
                let mid = indices.len() / 2;
                let (lefts, rights) = indices.split_at(mid);

                let mut lmin = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut lmax = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for &i in lefts { let c=centers[i]; let r=radii[i]; let h=Vector3::new(r,r,r); lmin=lmin.inf(&(c-h)); lmax=lmax.sup(&(c+h)); }
                let la = Aabb{min:lmin, max:lmax}.surface_area();

                let mut rmin = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut rmax = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for &i in rights { let c=centers[i]; let r=radii[i]; let h=Vector3::new(r,r,r); rmin=rmin.inf(&(c-h)); rmax=rmax.sup(&(c+h)); }
                let ra = Aabb{min:rmin, max:rmax}.surface_area();

                let cost = if parent_area > 0.0 { (la + ra) / parent_area * indices.len() as f64 } else { f64::INFINITY };

                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                }
            }

            indices.sort_by(|&a, &b| centers[a][best_axis].partial_cmp(&centers[b][best_axis]).unwrap());
            let mid = indices.len() / 2;
            let (left_part, right_part) = indices.split_at_mut(mid.max(1));

            let left = build(nodes, centers, radii, left_part);
            let right = build(nodes, centers, radii, right_part);

            let merged = nodes[left].aabb.merge(&nodes[right].aabb);
            nodes.push(BvhNode { aabb: merged, left: Some(left), right: Some(right), entity_index: None });
            nodes.len() - 1
        }

        let mut indices: Vec<usize> = (0..centers.len()).collect();
        let root = build(&mut nodes, centers, radii, &mut indices);
        Self { nodes, root }
    }

    pub fn refit(&mut self, centers: &[Vector3<f64>], radii: &[f64]) {
        for node in &mut self.nodes {
            if let Some(idx) = node.entity_index {
                if idx < centers.len() {
                    node.aabb = Aabb::from_center_and_radius(centers[idx], radii[idx]);
                }
            }
        }
        for i in (0..self.nodes.len()).rev() {
            let node = &self.nodes[i];
            if node.entity_index.is_none() {
                if let (Some(l), Some(r)) = (node.left, node.right) {
                    self.nodes[i].aabb = self.nodes[l].aabb.merge(&self.nodes[r].aabb);
                }
            }
        }
    }

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

    /// Ray traversal
    pub fn traverse_ray(&self, origin: Vector3<f64>, direction: Vector3<f64>) -> Vec<usize> {
        let mut result = Vec::new();
        self.traverse_ray_rec(self.root, origin, direction, &mut result);
        result
    }

    fn traverse_ray_rec(&self, idx: usize, origin: Vector3<f64>, dir: Vector3<f64>, result: &mut Vec<usize>) {
        if idx >= self.nodes.len() { return; }
        let node = &self.nodes[idx];
        if !node.aabb.intersects_ray(origin, dir) { return; }
        if let Some(e) = node.entity_index { result.push(e); }
        else {
            if let Some(l) = node.left { self.traverse_ray_rec(l, origin, dir, result); }
            if let Some(r) = node.right { self.traverse_ray_rec(r, origin, dir, result); }
        }
    }

    pub fn root_aabb(&self) -> Option<Aabb> {
        self.nodes.get(self.root).map(|n| n.aabb)
    }
}

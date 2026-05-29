//! Bounding Volume Hierarchy (BVH) with Refitting + SAH Construction
//!
//! High-quality top-down construction using Surface Area Heuristic.

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
    pub fn from_spheres(centers: &[Vector3<f64>], radii: &[f64]) -> Self {
        // Simple pairing construction (kept for compatibility)
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
                if chunk.len() == 1 { next_level.push(chunk[0]); continue; }
                let l = chunk[0]; let r = chunk[1];
                let merged = nodes[l].aabb.merge(&nodes[r].aabb);
                let p = nodes.len();
                nodes.push(BvhNode { aabb: merged, left: Some(l), right: Some(r), entity_index: None });
                next_level.push(p);
            }
            current_level = next_level;
        }
        Self { nodes, root: *current_level.first().unwrap_or(&0) }
    }

    /// Top-down SAH construction (improved quality)
    pub fn from_spheres_top_down(centers: &[Vector3<f64>], radii: &[f64]) -> Self {
        assert_eq!(centers.len(), radii.len());
        let mut nodes: Vec<BvhNode> = Vec::new();

        fn sah_cost(left_count: usize, right_count: usize, left_area: f64, right_area: f64, parent_area: f64) -> f64 {
            if parent_area <= 0.0 { return f64::INFINITY; }
            let p_left = left_area / parent_area;
            let p_right = right_area / parent_area;
            2.0 * (p_left * left_count as f64 + p_right * right_count as f64)
        }

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
            let mut parent_min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut parent_max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &i in indices.iter() {
                let c = centers[i]; let r = radii[i];
                let half = Vector3::new(r, r, r);
                parent_min = parent_min.inf(&(c - half));
                parent_max = parent_max.sup(&(c + half));
            }
            let parent_aabb = Aabb { min: parent_min, max: parent_max };
            let parent_area = parent_aabb.surface_area();

            // Evaluate SAH for each axis
            let mut best_axis = 0;
            let mut best_cost = f64::INFINITY;
            let mut best_split = indices.len() / 2;

            for axis in 0..3 {
                indices.sort_by(|&a, &b| centers[a][axis].partial_cmp(&centers[b][axis]).unwrap());

                // Evaluate median split cost (simple SAH approximation)
                let mid = indices.len() / 2;
                let (left_idx, right_idx) = indices.split_at(mid);

                let mut left_min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut left_max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for &i in left_idx {
                    let c = centers[i]; let r = radii[i];
                    let half = Vector3::new(r, r, r);
                    left_min = left_min.inf(&(c - half));
                    left_max = left_max.sup(&(c + half));
                }
                let left_area = Aabb { min: left_min, max: left_max }.surface_area();

                let mut right_min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
                let mut right_max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for &i in right_idx {
                    let c = centers[i]; let r = radii[i];
                    let half = Vector3::new(r, r, r);
                    right_min = right_min.inf(&(c - half));
                    right_max = right_max.sup(&(c + half));
                }
                let right_area = Aabb { min: right_min, max: right_max }.surface_area();

                let cost = sah_cost(left_idx.len(), right_idx.len(), left_area, right_area, parent_area);

                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis;
                    best_split = mid;
                }
            }

            // Final sort on best axis
            indices.sort_by(|&a, &b| centers[a][best_axis].partial_cmp(&centers[b][best_axis]).unwrap());

            let mid = best_split.max(1).min(indices.len() - 1);
            let (left_part, right_part) = indices.split_at_mut(mid);

            let left_child = build(nodes, centers, radii, left_part);
            let right_child = build(nodes, centers, radii, right_part);

            let merged = nodes[left_child].aabb.merge(&nodes[right_child].aabb);
            nodes.push(BvhNode {
                aabb: merged,
                left: Some(left_child),
                right: Some(right_child),
                entity_index: None,
            });

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
        self.traverse(self.root, query, &mut result);
        result
    }

    fn traverse(&self, idx: usize, query: &Aabb, result: &mut Vec<usize>) {
        if idx >= self.nodes.len() { return; }
        let node = &self.nodes[idx];
        if !node.aabb.intersects(query) { return; }
        if let Some(e) = node.entity_index {
            result.push(e);
        } else {
            if let Some(l) = node.left { self.traverse(l, query, result); }
            if let Some(r) = node.right { self.traverse(r, query, result); }
        }
    }

    pub fn query_intersecting_sphere(&self, center: Vector3<f64>, radius: f64) -> Vec<usize> {
        self.query_intersecting_aabb(&Aabb::from_center_and_radius(center, radius))
    }

    pub fn root_aabb(&self) -> Option<Aabb> {
        self.nodes.get(self.root).map(|n| n.aabb)
    }
}

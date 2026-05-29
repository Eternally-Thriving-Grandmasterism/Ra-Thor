//! Bounding Volume Hierarchy (BVH) with Refitting Support
//!
//! Supports high-quality construction and efficient queries + refitting.

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
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
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

                let left_idx = chunk[0];
                let right_idx = chunk[1];
                let merged = nodes[left_idx].aabb.merge(&nodes[right_idx].aabb);

                let parent_idx = nodes.len();
                nodes.push(BvhNode {
                    aabb: merged,
                    left: Some(left_idx),
                    right: Some(right_idx),
                    entity_index: None,
                });
                next_level.push(parent_idx);
            }
            current_level = next_level;
        }

        let root = *current_level.first().unwrap_or(&0);
        Self { nodes, root }
    }

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

            let mut min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

            for &i in indices.iter() {
                let c = centers[i];
                let r = radii[i];
                let half = Vector3::new(r, r, r);
                min = min.inf(&(c - half));
                max = max.sup(&(c + half));
            }

            let size = max - min;
            let axis = if size.x >= size.y && size.x >= size.z { 0 }
                       else if size.y >= size.z { 1 }
                       else { 2 };

            indices.sort_by(|&a, &b| {
                centers[a][axis].partial_cmp(&centers[b][axis]).unwrap()
            });

            let mid = indices.len() / 2;
            let (left_part, right_part) = indices.split_at_mut(mid);

            let left_child = build(nodes, centers, radii, left_part);
            let right_child = build(nodes, centers, radii, right_part);

            let merged_aabb = nodes[left_child].aabb.merge(&nodes[right_child].aabb);

            nodes.push(BvhNode {
                aabb: merged_aabb,
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
                if let (Some(left), Some(right)) = (node.left, node.right) {
                    self.nodes[i].aabb = self.nodes[left].aabb.merge(&self.nodes[right].aabb);
                }
            }
        }
    }

    /// Returns indices of entities whose AABBs intersect the query AABB.
    pub fn query_intersecting_aabb(&self, query: &Aabb) -> Vec<usize> {
        let mut result = Vec::new();
        self.traverse_intersecting(self.root, query, &mut result);
        result
    }

    fn traverse_intersecting(&self, node_idx: usize, query: &Aabb, result: &mut Vec<usize>) {
        if node_idx >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_idx];

        if !node.aabb.intersects(query) {
            return;
        }

        if let Some(entity_idx) = node.entity_index {
            result.push(entity_idx);
        } else {
            if let Some(left) = node.left {
                self.traverse_intersecting(left, query, result);
            }
            if let Some(right) = node.right {
                self.traverse_intersecting(right, query, result);
            }
        }
    }

    /// Convenience: Query entities intersecting a sphere (center + radius).
    pub fn query_intersecting_sphere(&self, center: Vector3<f64>, radius: f64) -> Vec<usize> {
        let query_aabb = Aabb::from_center_and_radius(center, radius);
        self.query_intersecting_aabb(&query_aabb)
    }

    pub fn root_aabb(&self) -> Option<Aabb> {
        self.nodes.get(self.root).map(|n| n.aabb)
    }
}

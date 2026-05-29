//! Bounding Volume Hierarchy (BVH) with Refitting Support
//!
//! Supports both bottom-up refitting and top-down construction.

use nalgebra::Vector3;
use crate::powrush::cga_primitives::CgaPoint;

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

    pub fn center(&self) -> Vector3<f64> {
        (self.min + self.max) * 0.5
    }

    pub fn surface_area(&self) -> f64 {
        let size = self.max - self.min;
        2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
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
    /// Simple construction (pairing).
    pub fn from_spheres(centers: &[Vector3<f64>], radii: &[f64]) -> Self {
        assert_eq!(centers.len(), radii.len());
        let mut nodes = Vec::new();

        for i in 0..centers.len() {
            let aabb = Aabb::from_center_and_radius(centers[i], radii[i]);
            nodes.push(BvhNode {
                aabb,
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

                let parent = nodes.len();
                nodes.push(BvhNode {
                    aabb: merged,
                    left: Some(left_idx),
                    right: Some(right_idx),
                    entity_index: None,
                });
                next_level.push(parent);
            }
            current_level = next_level;
        }

        let root = *current_level.first().unwrap_or(&0);
        Self { nodes, root }
    }

    /// Top-down construction using median split on longest axis.
    /// Produces better balanced trees than simple pairing.
    pub fn from_spheres_top_down(centers: &[Vector3<f64>], radii: &[f64]) -> Self {
        assert_eq!(centers.len(), radii.len());

        // Create all leaf nodes first
        let mut nodes: Vec<BvhNode> = centers
            .iter()
            .zip(radii.iter())
            .map(|(&center, &radius)| {
                let aabb = Aabb::from_center_and_radius(center, radius);
                BvhNode {
                    aabb,
                    left: None,
                    right: None,
                    entity_index: None, // will be set later
                }
            })
            .collect();

        // We need to keep track of which entity indices belong to which leaves
        // For simplicity, we'll rebuild indices during recursion

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

            // Compute AABB of current set
            let mut min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

            for &i in indices.iter() {
                let c = centers[i];
                let r = radii[i];
                min = min.inf(&(c - Vector3::new(r, r, r)));
                max = max.sup(&(c + Vector3::new(r, r, r)));
            }

            let current_aabb = Aabb { min, max };

            // Find longest axis
            let size = max - min;
            let axis = if size.x > size.y && size.x > size.z {
                0
            } else if size.y > size.z {
                1
            } else {
                2
            };

            // Sort indices by center on that axis
            indices.sort_by(|&a, &b| {
                centers[a][axis].partial_cmp(&centers[b][axis]).unwrap()
            });

            let mid = indices.len() / 2;
            let (left_indices, right_indices) = indices.split_at_mut(mid);

            let left_child = build(nodes, centers, radii, left_indices);
            let right_child = build(nodes, centers, radii, right_indices);

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

    /// Bottom-up refitting.
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

    pub fn root_aabb(&self) -> Option<Aabb> {
        self.nodes.get(self.root).map(|n| n.aabb)
    }
}

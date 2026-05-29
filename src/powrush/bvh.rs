//! Bounding Volume Hierarchy (BVH) with Refitting Support
//!
//! This module provides a basic BVH implementation focused on
//! efficient refitting for dynamic scenes (moving entities).
//! Designed to work alongside the CGA primitives and entity system.

use nalgebra::Vector3;
use crate::powrush::cga_primitives::CgaPoint;

/// A simple axis-aligned bounding box used in the BVH.
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

/// A node in the BVH.
#[derive(Debug, Clone)]
pub struct BvhNode {
    pub aabb: Aabb,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub entity_index: Option<usize>, // leaf node
}

/// A simple BVH focused on refitting.
#[derive(Debug, Clone)]
pub struct Bvh {
    pub nodes: Vec<BvhNode>,
    pub root: usize,
}

impl Bvh {
    /// Creates a new BVH from a list of bounding spheres (center + radius).
    /// This is a very basic construction for demonstration.
    pub fn from_spheres(centers: &[Vector3<f64>], radii: &[f64]) -> Self {
        assert_eq!(centers.len(), radii.len());

        let mut nodes = Vec::new();

        // Create leaf nodes
        for i in 0..centers.len() {
            let aabb = Aabb::from_center_and_radius(centers[i], radii[i]);
            nodes.push(BvhNode {
                aabb,
                left: None,
                right: None,
                entity_index: Some(i),
            });
        }

        // Very simple tree construction (not optimal)
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

                let merged_aabb = nodes[left_idx].aabb.merge(&nodes[right_idx].aabb);

                let parent_idx = nodes.len();
                nodes.push(BvhNode {
                    aabb: merged_aabb,
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

    /// Refits the BVH bottom-up.
    /// This is the key strategy for dynamic scenes.
    /// Call this after entities have moved.
    pub fn refit(&mut self, centers: &[Vector3<f64>], radii: &[f64]) {
        // Update leaf nodes first
        for node in &mut self.nodes {
            if let Some(idx) = node.entity_index {
                if idx < centers.len() {
                    node.aabb = Aabb::from_center_and_radius(centers[idx], radii[idx]);
                }
            }
        }

        // Bottom-up update of internal nodes
        for i in (0..self.nodes.len()).rev() {
            let node = &self.nodes[i];
            if node.entity_index.is_none() {
                if let (Some(left), Some(right)) = (node.left, node.right) {
                    let merged = self.nodes[left].aabb.merge(&self.nodes[right].aabb);
                    self.nodes[i].aabb = merged;
                }
            }
        }
    }

    /// Returns the root AABB of the hierarchy.
    pub fn root_aabb(&self) -> Option<Aabb> {
        self.nodes.get(self.root).map(|n| n.aabb)
    }
}

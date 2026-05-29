//! CgaHealingField with BVH acceleration support (preparatory integration)

use crate::powrush::cga_entity::CgaEntity;
use crate::powrush::cga_primitives::{Motor, CgaSphere};
use crate::powrush::bvh::Bvh;
use nalgebra::Vector3;

#[derive(Debug, Clone)]
pub struct CgaHealingField {
    pub name: String,
    pub entities: Vec<CgaEntity>,
    pub bvh: Option<Bvh>,           // Optional BVH for acceleration
    pub bvh_centers: Vec<Vector3<f64>>,
    pub bvh_radii: Vec<f64>,
}

impl CgaHealingField {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entities: Vec::new(),
            bvh: None,
            bvh_centers: vec![],
            bvh_radii: vec![],
        }
    }

    pub fn add_entity(&mut self, entity: CgaEntity) {
        self.entities.push(entity);
    }

    /// Rebuilds BVH from current entities (call after significant movement)
    pub fn rebuild_bvh(&mut self) {
        self.bvh_centers.clear();
        self.bvh_radii.clear();

        for e in &self.entities {
            let pos = e.world_position().to_euclidean();
            self.bvh_centers.push(pos);
            self.bvh_radii.push(1.0); // placeholder radius
        }

        if !self.bvh_centers.is_empty() {
            self.bvh = Some(Bvh::from_spheres_top_down(&self.bvh_centers, &self.bvh_radii));
        }
    }

    pub fn heal_entities_in_sphere(
        &mut self,
        healing_sphere: &CgaSphere,
        healing_motor: &Motor,
        mercy: f64,
        progress: f64,
    ) {
        // If BVH is available, we could query it here for acceleration
        for entity in &mut self.entities {
            if entity.intersects_sphere(healing_sphere) {
                entity.smooth_heal(healing_motor, progress, mercy);
            }
        }
    }

    // ... other methods ...
}

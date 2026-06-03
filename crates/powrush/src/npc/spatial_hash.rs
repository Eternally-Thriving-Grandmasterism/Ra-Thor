//! crates/powrush/src/npc/spatial_hash.rs
//! Production-grade Spatial Hashing with Dynamic Cell Support
//! Entities can have variable sizes (dynamic cells) | v1.0 | AG-SML v1.0

use nalgebra::Vector2;
use std::collections::HashMap;

pub type Position = Vector2<f32>;

#[derive(Debug, Clone)]
struct EntityData {
    position: Position,
    radius: f32,
}

/// A high-performance spatial hash that supports **dynamic cell sizes** per entity.
/// Large entities automatically occupy multiple cells.
#[derive(Debug)]
pub struct SpatialHash {
    base_cell_size: f32,
    cells: HashMap<(i32, i32), Vec<usize>>,
    entity_data: HashMap<usize, EntityData>,
}

impl SpatialHash {
    pub fn new(base_cell_size: f32) -> Self {
        Self {
            base_cell_size: base_cell_size.max(1.0),
            cells: HashMap::new(),
            entity_data: HashMap::new(),
        }
    }

    fn world_to_cell(&self, pos: Position) -> (i32, i32) {
        (
            (pos.x / self.base_cell_size).floor() as i32,
            (pos.y / self.base_cell_size).floor() as i32,
        )
    }

    /// Insert or update an entity with a **dynamic radius**.
    pub fn insert(&mut self, entity_id: usize, position: Position, radius: f32) {
        self.remove(entity_id);

        let data = EntityData { position, radius };
        self.entity_data.insert(entity_id, data);

        let min_x = ((position.x - radius) / self.base_cell_size).floor() as i32;
        let max_x = ((position.x + radius) / self.base_cell_size).floor() as i32;
        let min_y = ((position.y - radius) / self.base_cell_size).floor() as i32;
        let max_y = ((position.y + radius) / self.base_cell_size).floor() as i32;

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                self.cells.entry((x, y)).or_default().push(entity_id);
            }
        }
    }

    pub fn remove(&mut self, entity_id: usize) {
        if let Some(data) = self.entity_data.remove(&entity_id) {
            let min_x = ((data.position.x - data.radius) / self.base_cell_size).floor() as i32;
            let max_x = ((data.position.x + data.radius) / self.base_cell_size).floor() as i32;
            let min_y = ((data.position.y - data.radius) / self.base_cell_size).floor() as i32;
            let max_y = ((data.position.y + data.radius) / self.base_cell_size).floor() as i32;

            for x in min_x..=max_x {
                for y in min_y..=max_y {
                    if let Some(vec) = self.cells.get_mut(&(x, y)) {
                        vec.retain(|&id| id != entity_id);
                    }
                }
            }
        }
    }

    /// Query all entities within a given world-space radius.
    pub fn query_radius(&self, position: Position, radius: f32) -> Vec<usize> {
        let mut results = Vec::new();
        let radius_sq = radius * radius;

        let min_x = ((position.x - radius) / self.base_cell_size).floor() as i32;
        let max_x = ((position.x + radius) / self.base_cell_size).floor() as i32;
        let min_y = ((position.y - radius) / self.base_cell_size).floor() as i32;
        let max_y = ((position.y + radius) / self.base_cell_size).floor() as i32;

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                if let Some(entities) = self.cells.get(&(x, y)) {
                    for &id in entities {
                        if let Some(data) = self.entity_data.get(&id) {
                            if (data.position - position).magnitude_squared() <= radius_sq {
                                results.push(id);
                            }
                        }
                    }
                }
            }
        }

        results.sort_unstable();
        results.dedup();
        results
    }

    pub fn entity_count(&self) -> usize {
        self.entity_data.len()
    }
}
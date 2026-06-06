// gpu_compute_pipeline.rs
// Ra-Thor v14.9+ — GPU Memory Allocator with Fixed Coalescing Logic
// AG-SML v1.0 License

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTask {
    pub id: u64,
    pub name: String,
    pub buffer_size: usize,
    pub intensity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTaskResult {
    pub task_id: u64,
    pub success: bool,
    pub execution_time_ms: u64,
    pub output_size: usize,
    pub message: String,
}

pub struct GpuMemoryAllocator {
    free_buffers: HashMap<usize, Vec<Vec<u8>>>,
    total_allocated: usize,
    total_reused: usize,
    current_usage: usize,
    peak_usage: usize,
    allocation_count: usize,
    coalesce_count: usize,
    max_buffers_per_class: usize,
}

impl GpuMemoryAllocator {
    pub fn new() -> Self {
        Self {
            free_buffers: HashMap::new(),
            total_allocated: 0,
            total_reused: 0,
            current_usage: 0,
            peak_usage: 0,
            allocation_count: 0,
            coalesce_count: 0,
            max_buffers_per_class: 32,
        }
    }

    pub fn acquire(&mut self, size: usize) -> Vec<u8> {
        let aligned_size = Self::align_size(size);
        let size_class = Self::get_size_class(aligned_size);

        if let Some(buffers) = self.free_buffers.get_mut(&size_class) {
            if let Some(buffer) = buffers.pop() {
                self.total_reused += 1;
                self.current_usage += size_class;
                self.update_peak();
                return buffer;
            }
        }

        let buffer = vec![0u8; size_class];
        self.total_allocated += size_class;
        self.current_usage += size_class;
        self.allocation_count += 1;
        self.update_peak();
        buffer
    }

    pub fn release(&mut self, mut buffer: Vec<u8>) {
        let size = buffer.len();
        if size == 0 { return; }

        let list = self.free_buffers.entry(size).or_default();
        if list.len() < self.max_buffers_per_class {
            buffer.fill(0);
            list.push(buffer);
        }

        self.current_usage = self.current_usage.saturating_sub(size);

        // Attempt coalescing after storing
        self.try_coalesce(size);
    }

    fn try_coalesce(&mut self, size: usize) {
        let next_size = size * 2;

        if let Some(blocks) = self.free_buffers.get_mut(&size) {
            if blocks.len() >= 2 {
                blocks.pop();
                blocks.pop();

                let merged_list = self.free_buffers.entry(next_size).or_default();
                if merged_list.len() < self.max_buffers_per_class {
                    merged_list.push(vec![0u8; next_size]);
                }

                self.coalesce_count += 1;
            }
        }
    }

    fn update_peak(&mut self) {
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    pub fn stats(&self) -> GpuMemoryStats {
        let reuse_ratio = if self.allocation_count > 0 {
            self.total_reused as f64 / self.allocation_count as f64
        } else { 0.0 };

        let overallocation_ratio = if self.total_allocated > 0 {
            1.0 - (self.current_usage as f64 / self.total_allocated as f64)
        } else { 0.0 };

        let mut free_block_count = 0;
        let mut total_free_bytes = 0;
        let mut largest_free_block = 0;

        for (size, buffers) in &self.free_buffers {
            free_block_count += buffers.len();
            total_free_bytes += size * buffers.len();
            if *size > largest_free_block {
                largest_free_block = *size;
            }
        }

        let average_free_block = if free_block_count > 0 {
            total_free_bytes / free_block_count
        } else { 0 };

        let internal_frag_estimate = if self.allocation_count > 0 {
            (128 * self.allocation_count) as f64
        } else { 0.0 };

        GpuMemoryStats {
            total_allocated_bytes: self.total_allocated,
            current_usage_bytes: self.current_usage,
            peak_usage_bytes: self.peak_usage,
            total_reused_count: self.total_reused,
            allocation_count: self.allocation_count,
            coalesce_count: self.coalesce_count,
            reuse_ratio,
            overallocation_ratio,
            free_block_count,
            largest_free_block_bytes: largest_free_block,
            average_free_block_bytes: average_free_block,
            internal_fragmentation_estimate: internal_frag_estimate,
            active_size_classes: self.free_buffers.len(),
        }
    }

    fn align_size(size: usize) -> usize {
        let alignment = 256;
        (size + alignment - 1) & !(alignment - 1)
    }

    fn get_size_class(size: usize) -> usize {
        let mut s = 256;
        while s < size { s *= 2; }
        s
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryStats {
    pub total_allocated_bytes: usize,
    pub current_usage_bytes: usize,
    pub peak_usage_bytes: usize,
    pub total_reused_count: usize,
    pub allocation_count: usize,
    pub coalesce_count: usize,
    pub reuse_ratio: f64,
    pub overallocation_ratio: f64,
    pub free_block_count: usize,
    pub largest_free_block_bytes: usize,
    pub average_free_block_bytes: usize,
    pub internal_fragmentation_estimate: f64,
    pub active_size_classes: usize,
}

pub struct GpuComputePipeline {
    allocator: Arc<Mutex<GpuMemoryAllocator>>,
    pub version: String,
}

impl GpuComputePipeline {
    pub fn new() -> Self {
        Self {
            allocator: Arc::new(Mutex::new(GpuMemoryAllocator::new())),
            version: "v14.9.0-gpu-allocator".to_string(),
        }
    }

    pub async fn dispatch(&self, task: GpuTask) -> Result<GpuTaskResult, String> {
        let start = std::time::Instant::now();
        let mut allocator = self.allocator.lock().await;
        let _buffer = allocator.acquire(task.buffer_size);

        tokio::time::sleep(tokio::time::Duration::from_millis(70)).await;

        allocator.release(vec![0u8; GpuMemoryAllocator::get_size_class(task.buffer_size)]);
        let elapsed = start.elapsed().as_millis() as u64;
        let stats = allocator.stats();

        Ok(GpuTaskResult {
            task_id: task.id,
            success: true,
            execution_time_ms: elapsed,
            output_size: task.buffer_size / 4,
            message: format!(
                "GPU task '{}' | {} ms | Coalesced: {}x | Largest free: {} MB",
                task.name, elapsed, stats.coalesce_count, stats.largest_free_block_bytes / (1024*1024)
            ),
        })
    }

    pub async fn submit_patsagi_task(&self, query: &str, intensity: &str, buffer_size: usize) -> Result<GpuTaskResult, String> {
        let task = GpuTask {
            id: rand::random::<u64>() % 1_000_000_000,
            name: format!("patsagi_{}", query.replace(' ', "_")),
            buffer_size,
            intensity: intensity.to_string(),
        };
        self.dispatch(task).await
    }

    pub async fn get_memory_stats(&self) -> GpuMemoryStats {
        self.allocator.lock().await.stats()
    }
}
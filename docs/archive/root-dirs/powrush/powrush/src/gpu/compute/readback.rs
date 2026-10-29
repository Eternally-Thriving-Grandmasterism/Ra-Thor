//! Staging Buffer Pool + Async GPU Readback for Powrush-MMO (v17.6 Ultimate Production)
//!
//! Provides efficient, reusable staging buffers and safe asynchronous readback
//! from GPU storage buffers back to CPU. Critical for inspecting simulation
//! results (epigenetic fields, geometric harmony, NPC states, etc.) from
//! compute shaders without blocking the render thread.
//!
//! Features:
//! - StagingBufferPool with size-based reuse
//! - Async readback using wgpu map_async + staging buffer copy
//! - Integration helpers for Bevy + wgpu render pipelines
//! - Proper error handling and resource cleanup
//!
//! Designed to work alongside `pipeline.rs` and `mod.rs` in this module.
//!
//! All under AG-SML v1.0 • TOLC 8 Mercy Lattice • 7 Living Mercy Gates

use std::collections::HashMap;
use std::sync::Arc;

use bevy::render::render_resource::{Buffer, BufferDescriptor, BufferUsages, BufferView};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use wgpu::{BufferViewMut, Maintain};

/// A reusable staging buffer entry in the pool.
#[derive(Debug)]
struct StagingBuffer {
    buffer: Buffer,
    size: u64,
}

/// Production-grade staging buffer pool with reuse.
///
/// Reuses buffers of similar sizes to reduce allocation overhead during
/// frequent GPU ↔ CPU readbacks (e.g. epigenetic / geometric simulation results).
#[derive(Resource, Default)]
pub struct StagingBufferPool {
    buffers: HashMap<u64, Vec<StagingBuffer>>,
}

impl StagingBufferPool {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Get a staging buffer of at least the requested size.
    /// Reuses an existing buffer when possible.
    pub fn get_or_create(
        &mut self,
        render_device: &RenderDevice,
        size: u64,
        label: Option<&str>,
    ) -> Buffer {
        let entry = self.buffers.entry(size).or_default();

        if let Some(staging) = entry.pop() {
            return staging.buffer;
        }

        // Create new staging buffer
        render_device.create_buffer(&BufferDescriptor {
            label: label.or(Some("staging_buffer")),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a staging buffer to the pool for reuse.
    pub fn recycle(&mut self, buffer: Buffer, size: u64) {
        self.buffers
            .entry(size)
            .or_default()
            .push(StagingBuffer { buffer, size });
    }
}

/// Perform an asynchronous readback from a GPU buffer into a staging buffer.
///
/// This is the core production readback primitive. It:
/// 1. Copies `src_buffer` into a staging buffer
/// 2. Maps the staging buffer asynchronously
/// 3. Calls the provided callback with the mapped data (or error)
///
/// The caller is responsible for submitting the encoder and calling
/// `device.poll(Maintain::Wait)` or using Bevy's render graph scheduling.
pub async fn readback_buffer_async<F>(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    src_buffer: &Buffer,
    src_offset: u64,
    size: u64,
    pool: &mut StagingBufferPool,
    mut callback: F,
) where
    F: FnOnce(Result<BufferView, wgpu::BufferAsyncError>) + Send + 'static,
{
    let staging_buffer = pool.get_or_create(render_device, size, Some("readback_staging"));

    // Encode copy from src to staging
    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder"),
    });

    encoder.copy_buffer_to_buffer(src_buffer, src_offset, &staging_buffer, 0, size);
    render_queue.submit(std::iter::once(encoder.finish()));

    // Async map
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // In a real Bevy app this would be driven by the render graph.
    // For now we poll (acceptable in dedicated readback tasks).
    render_device.poll(Maintain::Wait);

    match receiver.receive().await {
        Some(Ok(())) => {
            let view = buffer_slice.get_mapped_range();
            callback(Ok(view));
            // Note: caller must unmap via drop or explicit unmap if keeping the buffer
            pool.recycle(staging_buffer, size);
        }
        Some(Err(e)) => {
            callback(Err(e));
            pool.recycle(staging_buffer, size);
        }
        None => {
            // Channel closed unexpectedly
            pool.recycle(staging_buffer, size);
        }
    }
}

/// Convenience synchronous readback (blocks until data is available).
/// Use sparingly in production; prefer the async version when possible.
pub fn readback_buffer_blocking(
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
    src_buffer: &Buffer,
    src_offset: u64,
    size: u64,
    pool: &mut StagingBufferPool,
) -> Result<Vec<u8>, wgpu::BufferAsyncError> {
    // This is a simplified blocking version for convenience / testing.
    // In real usage prefer the async path.
    let staging = pool.get_or_create(render_device, size, Some("readback_blocking"));

    let mut encoder = render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("blocking_readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(src_buffer, src_offset, &staging, 0, size);
    render_queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    render_device.poll(Maintain::Wait);

    match receiver.receive().now_or_never() {
        Some(Some(Ok(()))) => {
            let data = buffer_slice.get_mapped_range().to_vec();
            drop(buffer_slice);
            staging.unmap();
            pool.recycle(staging, size);
            Ok(data)
        }
        _ => {
            pool.recycle(staging, size);
            Err(wgpu::BufferAsyncError::DeviceLost)
        }
    }
}

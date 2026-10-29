//! Async GPU Readback with Optimized Staging Buffer Reuse (v16.9 Production)
//!
//! Optimized version of async GPU readback that reuses staging buffers
//! instead of allocating new ones on every readback.
//!
//! Benefits:
//! - Significantly reduced allocation overhead
//! - Better performance for frequent debug/simulation readbacks
//! - Lower memory fragmentation
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use bevy::render::render_resource::{Buffer, BufferUsages};
use std::collections::HashMap;
use wgpu::{BufferAsyncError, Maintain};

/// A simple staging buffer pool for reuse.
#[derive(Resource, Default)]
pub struct StagingBufferPool {
    /// Maps buffer size -> list of available staging buffers
    buffers: HashMap<u64, Vec<Buffer>>,
}

impl StagingBufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a staging buffer of the requested size (reuses if available).
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        size: u64,
    ) -> Buffer {
        if let Some(list) = self.buffers.get_mut(&size) {
            if let Some(buffer) = list.pop() {
                return buffer;
            }
        }

        // Create new staging buffer
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reusable_staging_buffer"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Return a staging buffer to the pool for reuse.
    pub fn return_buffer(&mut self, size: u64, buffer: Buffer) {
        self.buffers.entry(size).or_default().push(buffer);
    }
}

/// Optimized async readback that reuses staging buffers from the pool.
pub async fn read_buffer_async_reused(
    device: &wgpu::Device,
    pool: &mut StagingBufferPool,
    source_buffer: &Buffer,
    size: u64,
) -> Result<Vec<u8>, BufferAsyncError> {
    // Get reusable staging buffer
    let staging_buffer = pool.get_or_create(device, size);

    // Copy from source to staging
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback_encoder_reused"),
    });
    encoder.copy_buffer_to_buffer(source_buffer, 0, &staging_buffer, 0, size);

    let command_buffer = encoder.finish();
    device.queue().submit(Some(command_buffer));

    // Map asynchronously
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });

    device.poll(Maintain::Wait);
    receiver.await.expect("Failed to receive map result")?;

    let data = buffer_slice.get_mapped_range().to_vec();
    staging_buffer.unmap();

    // Return buffer to pool for reuse
    pool.return_buffer(size, staging_buffer);

    Ok(data)
}

/// Typed version with staging buffer reuse.
pub async fn read_buffer_as_vec_reused<T: bytemuck::Pod>(
    device: &wgpu::Device,
    pool: &mut StagingBufferPool,
    source_buffer: &Buffer,
    count: usize,
) -> Result<Vec<T>, BufferAsyncError> {
    let byte_size = (count * std::mem::size_of::<T>()) as u64;
    let bytes = read_buffer_async_reused(device, pool, source_buffer, byte_size).await?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

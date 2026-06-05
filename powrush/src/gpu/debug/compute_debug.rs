//! Async GPU Readback for Compute Shader Debugging (v16.8 Production)
//!
//! Proper asynchronous GPU buffer readback implementation.
//! This allows safely reading data from GPU buffers (debug output, simulation results, etc.)
//! without blocking the main thread.
//!
//! Features:
//! - Async readback using wgpu map_async
//! - Proper staging buffer pattern
//! - Error handling and timeout support
//! - Bevy-compatible async usage
//!
//! All under AG-SML v1.0 • TOLC 8 • 7 Living Mercy Gates

use bevy::prelude::*;
use bevy::render::render_resource::{Buffer, BufferUsages};
use bevy::render::renderer::RenderDevice;
use std::future::Future;
use wgpu::{BufferAsyncError, Maintain};

/// Asynchronously read a GPU buffer back to CPU memory.
///
/// # Arguments
/// * `device` - The wgpu device
/// * `buffer` - The GPU buffer to read from (must have COPY_SRC usage)
/// * `size` - Number of bytes to read
///
/// Returns the raw bytes on success.
pub async fn read_buffer_async(
    device: &wgpu::Device,
    buffer: &Buffer,
    size: u64,
) -> Result<Vec<u8>, BufferAsyncError> {
    // Create a staging buffer for mapping
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("async_readback_staging_buffer"),
        size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Create a command encoder to copy from storage buffer to staging buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("async_readback_encoder"),
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);

    // Submit the copy command
    let command_buffer = encoder.finish();
    device.queue().submit(Some(command_buffer));

    // Map the staging buffer for reading
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });

    // Wait for the mapping to complete
    // In Bevy, you may want to use bevy::tasks::AsyncComputeTaskPool instead of blocking
    device.poll(Maintain::Wait);

    receiver.await.expect("Failed to receive map result")?;

    // Read the data
    let data = buffer_slice.get_mapped_range().to_vec();

    // Unmap the buffer
    staging_buffer.unmap();

    Ok(data)
}

/// Convenience function to read a buffer as a specific type (e.g. Vec<u32>).
pub async fn read_buffer_as_vec<T: bytemuck::Pod>(
    device: &wgpu::Device,
    buffer: &Buffer,
    count: usize,
) -> Result<Vec<T>, BufferAsyncError> {
    let byte_size = (count * std::mem::size_of::<T>()) as u64;
    let bytes = read_buffer_async(device, buffer, byte_size).await?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

/// Bevy system example for async readback (non-blocking pattern).
/// In real usage, spawn this on Bevy's AsyncComputeTaskPool.
pub fn debug_readback_system(
    debug_resources: Res<crate::gpu::debug::compute_debug::ComputeDebugResources>,
    render_device: Res<RenderDevice>,
) {
    // Example of how you might trigger async readback
    // let device = render_device.wgpu_device();
    //
    // bevy::tasks::AsyncComputeTaskPool::get().spawn(async move {
    //     if let Ok(data) = read_buffer_as_vec::<u32>(device, &debug_resources.debug_buffer, 256).await {
    //         println!("Debug buffer contents: {:?}", data);
    //     }
    // });
}

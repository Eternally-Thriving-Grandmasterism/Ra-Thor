// gpu_compute_pipeline.rs
// Ra-Thor v14.9+ — GPU Memory Allocator + StagingBufferPool + Async Readback + Debug Utilities + Mercy-Gated Audit + Telemetry Consumers + Disk Persistence + Periodic Auto-Save + Graceful Shutdown + Signal Propagation + Error Handling + Retry Logic + Circuit Breaker + Runtime Config + Breaker Metrics + Prometheus Export
// AG-SML v1.0 License

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
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
    pub current_usage_bytes: self.current_usage,
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

// === StagingBufferPool ===
pub struct StagingBufferPool {
    allocator: GpuMemoryAllocator,
    staging_buffers: HashMap<usize, Vec<Vec<u8>>>,
}

impl StagingBufferPool {
    pub fn new() -> Self {
        Self {
            allocator: GpuMemoryAllocator::new(),
            staging_buffers: HashMap::new(),
        }
    }

    pub fn acquire_staging(&mut self, size: usize) -> Vec<u8> {
        let class = GpuMemoryAllocator::get_size_class(size);
        if let Some(buffers) = self.staging_buffers.get_mut(&class) {
            if let Some(buf) = buffers.pop() {
                return buf;
            }
        }
        self.allocator.acquire(size)
    }

    pub fn release_staging(&mut self, buffer: Vec<u8>) {
        let size = buffer.len();
        let class = GpuMemoryAllocator::get_size_class(size);
        let list = self.staging_buffers.entry(class).or_default();
        if list.len() < 16 {
            list.push(buffer);
        } else {
            self.allocator.release(buffer);
        }
    }
}

// === Async Readback simulation ===
pub async fn readback_buffer_async(buffer: Vec<u8>) -> Result<Vec<u8>, String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
    Ok(buffer)
}

pub fn readback_buffer_blocking(buffer: Vec<u8>) -> Result<Vec<u8>, String> {
    Ok(buffer)
}

// === Debug utilities ===
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugOutputBuffer {
    pub label: String,
    pub data: Vec<u8>,
    pub timestamp_ms: u64,
}

impl DebugOutputBuffer {
    pub fn new(label: &str, data: Vec<u8>) -> Self {
        Self {
            label: label.to_string(),
            data,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn inspect(&self) -> String {
        format!("DEBUG [{} @ {}ms]: {} bytes | first 16: {:02x?}", 
            self.label, self.timestamp_ms, self.data.len(), 
            &self.data[..std::cmp::min(16, self.data.len())])
    }
}

// === Mercy Telemetry + Persistence + Auto-Save + Graceful Shutdown + Signal Propagation + Error Handling + Retry Logic + Circuit Breaker + Runtime Config + Breaker Metrics + Prometheus Export ===
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MercyTelemetry {
    pub total_audits: u64,
    pub total_council_ready: u64,
    pub sum_mercy_norm: f64,
    pub min_mercy_norm: f64,
    pub max_mercy_norm: f64,
    pub last_mercy_norm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyTelemetrySummary {
    pub total_audits: u64,
    pub council_ready_ratio: f64,
    pub avg_mercy_norm: f64,
    pub min_mercy_norm: f64,
    pub max_mercy_norm: f64,
    pub last_mercy_norm: f64,
}

impl MercyTelemetry {
    pub fn new() -> Self {
        Self {
            min_mercy_norm: 1.0,
            max_mercy_norm: 0.0,
            ..Default::default()
        }
    }

    pub fn consume(&mut self, audit: &MercyGpuAudit) {
        self.total_audits += 1;
        if audit.council_ready {
            self.total_council_ready += 1;
        }
        self.sum_mercy_norm += audit.mercy_norm;
        if audit.mercy_norm < self.min_mercy_norm {
            self.min_mercy_norm = audit.mercy_norm;
        }
        if audit.mercy_norm > self.max_mercy_norm {
            self.max_mercy_norm = audit.mercy_norm;
        }
        self.last_mercy_norm = audit.mercy_norm;
    }

    pub fn summary(&self) -> MercyTelemetrySummary {
        let avg = if self.total_audits > 0 {
            self.sum_mercy_norm / self.total_audits as f64
        } else { 0.0 };
        let ready_ratio = if self.total_audits > 0 {
            self.total_council_ready as f64 / self.total_audits as f64
        } else { 0.0 };

        MercyTelemetrySummary {
            total_audits: self.total_audits,
            council_ready_ratio: ready_ratio,
            avg_mercy_norm: avg,
            min_mercy_norm: self.min_mercy_norm,
            max_mercy_norm: self.max_mercy_norm,
            last_mercy_norm: self.last_mercy_norm,
        }
    }
}

/// Simple Circuit Breaker for telemetry persistence with runtime-configurable thresholds
pub struct TelemetryCircuitBreaker {
    failure_count: AtomicUsize,
    last_failure_time: std::sync::Mutex<Option<Instant>>,
    failure_threshold: AtomicUsize,
    reset_timeout: std::sync::Mutex<Duration>,
}

impl TelemetryCircuitBreaker {
    pub fn new(threshold: usize, reset_timeout: Duration) -> Self {
        Self {
            failure_count: AtomicUsize::new(0),
            last_failure_time: std::sync::Mutex::new(None),
            failure_threshold: AtomicUsize::new(threshold),
            reset_timeout: std::sync::Mutex::new(reset_timeout),
        }
    }

    pub fn is_open(&self) -> bool {
        if self.failure_count.load(Ordering::Relaxed) < self.failure_threshold.load(Ordering::Relaxed) {
            return false;
        }
        if let Ok(guard) = self.last_failure_time.lock() {
            if let Some(time) = *guard {
                if let Ok(timeout) = self.reset_timeout.lock() {
                    return time.elapsed() < *timeout;
                }
            }
        }
        false
    }

    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        if let Ok(mut guard) = self.last_failure_time.lock() {
            *guard = None;
        }
    }

    pub fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut guard) = self.last_failure_time.lock() {
            *guard = Some(Instant::now());
        }
    }

    /// Runtime configuration
    pub fn set_threshold(&self, threshold: usize) {
        self.failure_threshold.store(threshold, Ordering::Relaxed);
    }

    pub fn set_reset_timeout(&self, timeout: Duration) {
        if let Ok(mut guard) = self.reset_timeout.lock() {
            *guard = timeout;
        }
    }

    // === Metrics getters ===
    pub fn current_failure_count(&self) -> usize {
        self.failure_count.load(Ordering::Relaxed)
    }

    pub fn last_failure_instant(&self) -> Option<Instant> {
        self.last_failure_time.lock().ok().and_then(|g| *g)
    }

    pub fn remaining_cooldown(&self) -> Option<Duration> {
        if !self.is_open() {
            return None;
        }
        if let Ok(guard) = self.last_failure_time.lock() {
            if let Some(time) = *guard {
                if let Ok(timeout) = self.reset_timeout.lock() {
                    let elapsed = time.elapsed();
                    if elapsed < *timeout {
                        return Some(*timeout - elapsed);
                    }
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryBreakerMetrics {
    pub failure_count: usize,
    pub is_open: bool,
    pub threshold: usize,
    pub reset_timeout_secs: u64,
    pub last_failure_secs_ago: Option<u64>,
    pub remaining_cooldown_secs: Option<u64>,
}

pub struct GpuComputePipeline {
    allocator: Arc<Mutex<GpuMemoryAllocator>>,
    staging_pool: Arc<Mutex<StagingBufferPool>>,
    telemetry: Arc<Mutex<MercyTelemetry>>,
    telemetry_save_path: Option<String>,
    mercy_telemetry_shutdown: Arc<Mutex<Option<tokio::sync::broadcast::Sender<()>>>>,
    mercy_telemetry_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    telemetry_circuit_breaker: Arc<TelemetryCircuitBreaker>,
    telemetry_retry_count: AtomicUsize,
    pub version: String,
}

impl GpuComputePipeline {
    pub fn new() -> Self {
        Self {
            allocator: Arc::new(Mutex::new(GpuMemoryAllocator::new())),
            staging_pool: Arc::new(Mutex::new(StagingBufferPool::new())),
            telemetry: Arc::new(Mutex::new(MercyTelemetry::new())),
            telemetry_save_path: None,
            mercy_telemetry_shutdown: Arc::new(Mutex::new(None)),
            mercy_telemetry_handle: Arc::new(Mutex::new(None)),
            telemetry_circuit_breaker: Arc::new(TelemetryCircuitBreaker::new(5, Duration::from_secs(30))),
            telemetry_retry_count: AtomicUsize::new(3),
            version: "v14.9.12-gpu-mercy-prometheus".to_string(),
        }
    }

    pub fn set_mercy_telemetry_save_path(&mut self, path: impl Into<String>) {
        self.telemetry_save_path = Some(path.into());
    }

    /// Runtime configuration for resilience thresholds
    pub fn set_telemetry_retry_count(&self, count: usize) {
        self.telemetry_retry_count.store(count, Ordering::Relaxed);
    }

    pub fn set_telemetry_breaker_threshold(&self, threshold: usize) {
        self.telemetry_circuit_breaker.set_threshold(threshold);
    }

    pub fn set_telemetry_breaker_reset_timeout(&self, timeout: Duration) {
        self.telemetry_circuit_breaker.set_reset_timeout(timeout);
    }

    /// Breaker state metrics for observability (Lattice Conductor / dashboards)
    pub fn get_telemetry_breaker_metrics(&self) -> TelemetryBreakerMetrics {
        let breaker = &self.telemetry_circuit_breaker;
        let failure_count = breaker.current_failure_count();
        let is_open = breaker.is_open();
        let threshold = breaker.failure_threshold.load(Ordering::Relaxed);
        let reset_timeout_secs = breaker.reset_timeout.lock().map(|d| d.as_secs()).unwrap_or(30);

        let last_failure_secs_ago = breaker.last_failure_instant().map(|t| t.elapsed().as_secs());
        let remaining_cooldown_secs = breaker.remaining_cooldown().map(|d| d.as_secs());

        TelemetryBreakerMetrics {
            failure_count,
            is_open,
            threshold,
            reset_timeout_secs,
            last_failure_secs_ago,
            remaining_cooldown_secs,
        }
    }

    /// Export mercy telemetry breaker metrics in Prometheus text exposition format.
    /// Ready to be served from any /metrics HTTP endpoint.
    pub fn export_telemetry_breaker_prometheus(&self) -> String {
        let m = self.get_telemetry_breaker_metrics();
        let is_open = if m.is_open { 1 } else { 0 };
        let last_failure = m.last_failure_secs_ago.unwrap_or(0);
        let remaining = m.remaining_cooldown_secs.unwrap_or(0);

        format!(
            r#"# HELP ra_thor_mercy_telemetry_breaker_failure_count Current consecutive save failures
# TYPE ra_thor_mercy_telemetry_breaker_failure_count gauge
ra_thor_mercy_telemetry_breaker_failure_count {failure_count}

# HELP ra_thor_mercy_telemetry_breaker_is_open Whether the circuit breaker is currently open (1 = open)
# TYPE ra_thor_mercy_telemetry_breaker_is_open gauge
ra_thor_mercy_telemetry_breaker_is_open {is_open}

# HELP ra_thor_mercy_telemetry_breaker_threshold Configured failure threshold to open breaker
# TYPE ra_thor_mercy_telemetry_breaker_threshold gauge
ra_thor_mercy_telemetry_breaker_threshold {threshold}

# HELP ra_thor_mercy_telemetry_breaker_reset_timeout_seconds Configured reset timeout in seconds
# TYPE ra_thor_mercy_telemetry_breaker_reset_timeout_seconds gauge
ra_thor_mercy_telemetry_breaker_reset_timeout_seconds {reset_timeout_secs}

# HELP ra_thor_mercy_telemetry_breaker_last_failure_seconds_ago Seconds since last recorded failure
# TYPE ra_thor_mercy_telemetry_breaker_last_failure_seconds_ago gauge
ra_thor_mercy_telemetry_breaker_last_failure_seconds_ago {last_failure}

# HELP ra_thor_mercy_telemetry_breaker_remaining_cooldown_seconds Remaining seconds until breaker can attempt reset
# TYPE ra_thor_mercy_telemetry_breaker_remaining_cooldown_seconds gauge
ra_thor_mercy_telemetry_breaker_remaining_cooldown_seconds {remaining}
"#,
            failure_count = m.failure_count,
            is_open = is_open,
            threshold = m.threshold,
            reset_timeout_secs = m.reset_timeout_secs,
            last_failure = last_failure,
            remaining = remaining,
        )
    }

    pub async fn dispatch(&self, task: GpuTask) -> Result<GpuTaskResult, String> {
        let start = std::time::Instant::now();
        let mut allocator = self.allocator.lock().await;
        let mut staging = self.staging_pool.lock().await;

        let staging_buf = staging.acquire_staging(task.buffer_size);
        tokio::time::sleep(tokio::time::Duration::from_millis(70)).await;

        let readback = readback_buffer_async(staging_buf.clone()).await?;
        let debug = DebugOutputBuffer::new(&task.name, readback.clone());

        staging.release_staging(staging_buf);
        allocator.release(vec![0u8; GpuMemoryAllocator::get_size_class(task.buffer_size)]);

        let elapsed = start.elapsed().as_millis() as u64;
        let stats = allocator.stats();

        Ok(GpuTaskResult {
            task_id: task.id,
            success: true,
            execution_time_ms: elapsed,
            output_size: task.buffer_size / 4,
            message: format!(
                "GPU task '{}' | {} ms | Coalesced: {}x | Readback: {} bytes | Debug: {}",
                task.name, elapsed, stats.coalesce_count, readback.len(), debug.inspect()
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

    pub async fn consume_mercy_audit(&self, audit: &MercyGpuAudit) {
        let mut tel = self.telemetry.lock().await;
        tel.consume(audit);
    }

    pub async fn get_mercy_telemetry_summary(&self) -> MercyTelemetrySummary {
        let tel = self.telemetry.lock().await;
        tel.summary()
    }

    /// Internal save helper that respects runtime retry count + circuit breaker
    async fn save_with_retry_and_breaker(&self, path: impl AsRef<std::path::Path>, data: String) -> Result<(), String> {
        let breaker = &self.telemetry_circuit_breaker;
        let max_retries = self.telemetry_retry_count.load(Ordering::Relaxed);

        if breaker.is_open() {
            return Err("Telemetry circuit breaker is OPEN - skipping save".to_string());
        }

        for attempt in 0..=max_retries {
            match tokio::fs::write(&path, &data).await {
                Ok(_) => {
                    breaker.record_success();
                    return Ok(());
                }
                Err(e) if attempt < max_retries => {
                    let backoff_ms = 50 * (1 << attempt);
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                Err(e) => {
                    breaker.record_failure();
                    return Err(format!("Failed after {} retries: {}", max_retries, e));
                }
            }
        }
        breaker.record_failure();
        Err("Retry loop exit".to_string())
    }

    pub async fn save_mercy_telemetry(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let tel = self.telemetry.lock().await;
        let json = serde_json::to_string_pretty(&*tel)
            .map_err(|e| format!("Failed to serialize telemetry: {}", e))?;
        self.save_with_retry_and_breaker(path, json).await
    }

    pub async fn load_mercy_telemetry(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let data = tokio::fs::read_to_string(path).await
            .map_err(|e| format!("Failed to read telemetry file: {}", e))?;
        let loaded: MercyTelemetry = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to deserialize telemetry: {}", e))?;
        let mut tel = self.telemetry.lock().await;
        *tel = loaded;
        Ok(())
    }

    /// Start periodic auto-save (respects runtime config for retries + breaker)
    pub async fn start_periodic_mercy_telemetry_save(&self, interval_secs: u64) -> Result<(), String> {
        let path = self.telemetry_save_path.clone()
            .ok_or_else(|| "No telemetry save path configured. Call set_mercy_telemetry_save_path first.".to_string())?;
        let telemetry = self.telemetry.clone();
        let breaker = self.telemetry_circuit_breaker.clone();
        let retry_count = self.telemetry_retry_count.clone();

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

        {
            let mut guard = self.mercy_telemetry_shutdown.lock().await;
            *guard = Some(shutdown_tx.clone());
        }

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if breaker.is_open() {
                            continue;
                        }

                        let tel = telemetry.lock().await;
                        if let Ok(json) = serde_json::to_string_pretty(&*tel) {
                            let max_retries = retry_count.load(Ordering::Relaxed);
                            if let Err(e) = GpuComputePipeline::save_with_retry_and_breaker_static(&path, json, max_retries, &breaker).await {
                                eprintln!("[Ra-Thor] Periodic mercy telemetry save failed: {}", e);
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });

        {
            let mut guard = self.mercy_telemetry_handle.lock().await;
            *guard = Some(handle);
        }

        Ok(())
    }

    async fn save_with_retry_and_breaker_static(
        path: impl AsRef<std::path::Path>,
        data: String,
        max_retries: usize,
        breaker: &Arc<TelemetryCircuitBreaker>,
    ) -> Result<(), String> {
        if breaker.is_open() {
            return Err("Circuit breaker OPEN".to_string());
        }

        for attempt in 0..=max_retries {
            match tokio::fs::write(&path, &data).await {
                Ok(_) => {
                    breaker.record_success();
                    return Ok(());
                }
                Err(e) if attempt < max_retries => {
                    let backoff_ms = 50 * (1 << attempt);
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                Err(e) => {
                    breaker.record_failure();
                    return Err(format!("Failed after retries: {}", e));
                }
            }
        }
        breaker.record_failure();
        Err("Retry loop exit".to_string())
    }

    pub async fn shutdown_mercy_telemetry_auto_save(&self) -> Result<(), String> {
        let has_handle = {
            let guard = self.mercy_telemetry_handle.lock().await;
            guard.is_some()
        };

        if !has_handle {
            return Err("No mercy telemetry auto-save task is running".to_string());
        }

        {
            let guard = self.mercy_telemetry_shutdown.lock().await;
            if let Some(tx) = guard.as_ref() {
                let _ = tx.send(());
            }
        }

        let handle = {
            let mut guard = self.mercy_telemetry_handle.lock().await;
            guard.take()
        };

        if let Some(h) = handle {
            match tokio::time::timeout(std::time::Duration::from_secs(5), h).await {
                Ok(Ok(_)) => {
                    println!("[Ra-Thor] Mercy telemetry auto-save gracefully shut down.");
                    Ok(())
                }
                Ok(Err(e)) => Err(format!("Task panicked: {:?}", e)),
                Err(_) => Err("Shutdown timed out after 5s".to_string()),
            }
        } else {
            Err("Handle missing".to_string())
        }
    }
}

// === Mercy-Gated Audit Logic ===
pub const MERCY_NORM_THRESHOLD: f64 = 0.85;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MercyGpuAudit {
    pub task_id: u64,
    pub mercy_norm: f64,
    pub execution_time_ms: u64,
    pub reuse_ratio: f64,
    pub fragmentation_estimate: f64,
    pub council_ready: bool,
    pub trace: String,
}

impl MercyGpuAudit {
    pub fn is_council_ready(&self) -> bool {
        self.mercy_norm >= MERCY_NORM_THRESHOLD
    }

    pub fn suggested_confidence_delta(&self) -> f64 {
        (self.mercy_norm - 0.5) * 0.18
    }

    pub fn summary(&self) -> String {
        format!(
            "MercyGpuAudit | norm={:.4} | council_ready={} | time={}ms | reuse={:.2} | frag={:.1}",
            self.mercy_norm, self.is_council_ready(), self.execution_time_ms, self.reuse_ratio, self.fragmentation_estimate
        )
    }
}

pub fn calculate_mercy_norm(stats: &GpuMemoryStats, result: &GpuTaskResult, task: &GpuTask) -> f64 {
    let success_factor = if result.success { 1.0 } else { 0.25 };
    let efficiency = (1000.0 / (result.execution_time_ms as f64 + 1.0)).min(1.0);
    let reuse_factor = stats.reuse_ratio.clamp(0.0, 1.0);
    let waste = (stats.internal_fragmentation_estimate / 20000.0).min(0.3);
    let coalesce_bonus = (stats.coalesce_count as f64 / (stats.allocation_count as f64 + 1.0)).min(0.15);

    let norm = ((success_factor * 0.42) + (efficiency * 0.28) + (reuse_factor * 0.22) - waste + coalesce_bonus).clamp(0.0, 1.0);
    norm
}

impl GpuComputePipeline {
    pub async fn dispatch_with_mercy_audit(&self, task: GpuTask) -> Result<(GpuTaskResult, MercyGpuAudit), String> {
        let result = self.dispatch(task.clone()).await?;
        let stats = self.get_memory_stats().await;

        let mercy_norm = calculate_mercy_norm(&stats, &result, &task);

        let audit = MercyGpuAudit {
            task_id: result.task_id,
            mercy_norm,
            execution_time_ms: result.execution_time_ms,
            reuse_ratio: stats.reuse_ratio,
            fragmentation_estimate: stats.internal_fragmentation_estimate,
            council_ready: mercy_norm >= MERCY_NORM_THRESHOLD,
            trace: format!(
                "MERCY_AUDIT task='{}' norm={:.4} time={}ms reuse={:.2} frag={:.1} council_ready={}",
                task.name, mercy_norm, result.execution_time_ms, stats.reuse_ratio, stats.internal_fragmentation_estimate, mercy_norm >= MERCY_NORM_THRESHOLD
            ),
        };

        self.consume_mercy_audit(&audit).await;

        Ok((result, audit))
    }

    pub async fn submit_patsagi_task_with_audit(
        &self,
        query: &str,
        intensity: &str,
        buffer_size: usize,
    ) -> Result<(GpuTaskResult, MercyGpuAudit), String> {
        let task = GpuTask {
            id: rand::random::<u64>() % 1_000_000_000,
            name: format!("patsagi_{}", query.replace(' ', "_")),
            buffer_size,
            intensity: intensity.to_string(),
        };
        self.dispatch_with_mercy_audit(task).await
    }
}
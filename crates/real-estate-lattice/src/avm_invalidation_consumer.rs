//! Production-ready helper for running the Redis Streams AVM Invalidation Consumer.
//!
//! This module provides a clean, resilient way to run the consumer in production
//! with graceful shutdown, automatic restart on error, and good observability hooks.
//!
//! Usage:
//! ```ignore
//! let consumer = AvmInvalidationConsumer::new(
//!     redis_url,
//!     "avm:invalidation",
//!     "valuation-consumers",
//!     "pod-abc123",
//!     cache,
//! );
//!
//! consumer.run_with_shutdown(shutdown_rx).await;
//! ```

use std::sync::Arc;
use tokio::sync::{watch, Mutex};

#[cfg(feature = "redis")]
use crate::avm_cache_invalidation::RedisStreamConsumer;
use crate::valuation_confidence_scorer::AvmCache;

pub struct AvmInvalidationConsumer {
    #[cfg(feature = "redis")]
    inner: Option<RedisStreamConsumer>,
}

impl AvmInvalidationConsumer {
    #[cfg(feature = "redis")]
    pub fn new(
        redis_url: &str,
        stream: &str,
        group: &str,
        consumer_name: &str,
        cache: Arc<Mutex<AvmCache>>,
    ) -> Result<Self, redis::RedisError> {
        let inner = RedisStreamConsumer::new(redis_url, stream, group, consumer_name, cache)?;
        Ok(Self { inner: Some(inner) })
    }

    #[cfg(not(feature = "redis"))]
    pub fn new(
        _redis_url: &str,
        _stream: &str,
        _group: &str,
        _consumer_name: &str,
        _cache: Arc<Mutex<AvmCache>>,
    ) -> Result<Self, String> {
        Ok(Self { inner: None })
    }

    /// Run the consumer with graceful shutdown support.
    #[cfg(feature = "redis")]
    pub async fn run_with_shutdown(
        &self,
        mut shutdown: watch::Receiver<bool>,
    ) -> Result<(), redis::RedisError> {
        let consumer = self.inner.as_ref().unwrap();

        loop {
            tokio::select! {
                biased;

                _ = shutdown.changed() => {
                    println!("Shutdown signal received. Stopping AVM invalidation consumer...");
                    break;
                }

                result = consumer.run() => {
                    match result {
                        Ok(_) => break,
                        Err(e) => {
                            eprintln!("Consumer error: {:?}. Restarting in 5 seconds...", e);
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "redis"))]
    pub async fn run_with_shutdown(
        &self,
        _shutdown: watch::Receiver<bool>,
    ) -> Result<(), String> {
        println!("Redis feature not enabled. Consumer will not run.");
        Ok(())
    }
}

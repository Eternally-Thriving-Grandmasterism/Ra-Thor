//! Redis Pub/Sub based Invalidation for AVM Cache
//!
//! Provides distributed cache invalidation using Redis Pub/Sub.
//! When relevant events occur (new offers, status certificate updates, etc.),
//! we publish invalidation messages so all instances can evict stale AVM entries.
//!
//! **Architecture**:
//! - Publisher: Called from MultiOfferTrackEngine, StatusCertificateAnalyzer, etc.
//! - Subscriber: Long-running task that listens and invalidates local AvmCache
//! - Message format is simple JSON for easy interoperability
//!
//! This enables the Hybrid Valuation system to stay reasonably fresh across distributed deployments.
//!
//! To enable: Add `redis` crate and use the `redis` feature.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "redis")]
use redis::{Client, Commands, PubSubCommands};

use crate::valuation_confidence_scorer::AvmCache;

/// Message published when an AVM cache entry should be invalidated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvmInvalidationMessage {
    pub property_key: String,
    pub reason: String,           // e.g. "new_offer", "status_certificate_update", "developer_risk_change"
    pub timestamp: u64,
}

/// Publishes invalidation events to Redis.
#[cfg(feature = "redis")]
pub struct RedisInvalidationPublisher {
    client: Client,
    channel: String,
}

#[cfg(feature = "redis")]
impl RedisInvalidationPublisher {
    pub fn new(redis_url: &str, channel: &str) -> Result<Self, redis::RedisError> {
        let client = Client::open(redis_url)?;
        Ok(Self {
            client,
            channel: channel.to_string(),
        })
    }

    /// Publish an invalidation message for a specific property.
    pub fn publish_invalidation(
        &self,
        property_key: &str,
        reason: &str,
    ) -> Result<(), redis::RedisError> {
        let mut con = self.client.get_connection()?;

        let msg = AvmInvalidationMessage {
            property_key: property_key.to_string(),
            reason: reason.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let payload = serde_json::to_string(&msg).unwrap();
        con.publish(&self.channel, payload)?;
        Ok(())
    }
}

/// Subscribes to invalidation events and applies them to a local AvmCache.
#[cfg(feature = "redis")]
pub struct RedisInvalidationSubscriber {
    client: Client,
    channel: String,
    cache: Arc<tokio::sync::Mutex<AvmCache>>,
}

#[cfg(feature = "redis")]
impl RedisInvalidationSubscriber {
    pub fn new(
        redis_url: &str,
        channel: &str,
        cache: Arc<tokio::sync::Mutex<AvmCache>>,
    ) -> Result<Self, redis::RedisError> {
        let client = Client::open(redis_url)?;
        Ok(Self {
            client,
            channel: channel.to_string(),
            cache,
        })
    }

    /// Start listening for invalidation messages (blocking).
    /// Run this in a separate tokio task.
    pub async fn run(&self) -> Result<(), redis::RedisError> {
        let mut pubsub = self.client.get_async_connection().await?.into_pubsub();
        pubsub.subscribe(&self.channel).await?;

        let mut stream = pubsub.into_on_message();

        while let Some(msg) = stream.next().await {
            if let Ok(payload) = msg.get_payload::<String>() {
                if let Ok(invalidation) = serde_json::from_str::<AvmInvalidationMessage>(&payload) {
                    let mut cache = self.cache.lock().await;
                    // For simplicity we just remove the key. A more advanced version could mark as stale.
                    cache.entries.remove(&invalidation.property_key); // Note: requires making entries pub(crate) or adding a method
                    // In real code we would call a proper remove method on AvmCache
                }
            }
        }

        Ok(())
    }
}

// Fallback stubs when redis feature is not enabled
#[cfg(not(feature = "redis"))]
pub struct RedisInvalidationPublisher;

#[cfg(not(feature = "redis"))]
#[allow(dead_code)]
impl RedisInvalidationPublisher {
    pub fn new(_redis_url: &str, _channel: &str) -> Result<Self, String> {
        Err("redis feature not enabled".to_string())
    }
}

#[cfg(not(feature = "redis"))]
pub struct RedisInvalidationSubscriber;

// Note: In a full implementation we would expose a method on AvmCache like:
// pub fn remove(&mut self, key: &str)
// For now this module demonstrates the pattern and integration points.

//! Redis Streams based Invalidation for AVM Cache (Reliable Version)
//!
//! Upgraded from Pub/Sub to Redis Streams for durability, replayability,
//! and consumer group support.
//!
//! **Why Streams?**
//! - Messages are persisted until explicitly trimmed
//! - Consumer groups provide automatic load balancing and offset tracking
//! - Failed consumers can resume from last acknowledged message
//! - Much more reliable for distributed cache invalidation
//!
//! Recommended for production use of the Hybrid Valuation system.

use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(feature = "redis")]
use redis::{Client, Commands, streams};

use crate::valuation_confidence_scorer::AvmCache;

/// Invalidation message published to the stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvmInvalidationMessage {
    pub property_key: String,
    pub reason: String,
    pub timestamp: u64,
}

/// Publishes invalidation events to a Redis Stream.
#[cfg(feature = "redis")]
pub struct RedisStreamPublisher {
    client: Client,
    stream: String,
}

#[cfg(feature = "redis")]
impl RedisStreamPublisher {
    pub fn new(redis_url: &str, stream: &str) -> Result<Self, redis::RedisError> {
        let client = Client::open(redis_url)?;
        Ok(Self {
            client,
            stream: stream.to_string(),
        })
    }

    /// Publish an invalidation message to the stream.
    pub fn publish(&self, property_key: &str, reason: &str) -> Result<String, redis::RedisError> {
        let mut con = self.client.get_connection()?;

        let msg = AvmInvalidationMessage {
            property_key: property_key.to_string(),
            reason: reason.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let fields = vec![
            ("property_key", msg.property_key),
            ("reason", msg.reason),
            ("timestamp", msg.timestamp.to_string()),
        ];

        // XADD returns the message ID
        let message_id: String = con.xadd(&self.stream, "*", &fields)?;
        Ok(message_id)
    }
}

/// Consumes invalidation messages using Redis Streams consumer groups.
#[cfg(feature = "redis")]
pub struct RedisStreamConsumer {
    client: Client,
    stream: String,
    group: String,
    consumer_name: String,
    cache: Arc<tokio::sync::Mutex<AvmCache>>,
}

#[cfg(feature = "redis")]
impl RedisStreamConsumer {
    pub fn new(
        redis_url: &str,
        stream: &str,
        group: &str,
        consumer_name: &str,
        cache: Arc<tokio::sync::Mutex<AvmCache>>,
    ) -> Result<Self, redis::RedisError> {
        let client = Client::open(redis_url)?;

        // Create consumer group if it doesn't exist
        let mut con = client.get_connection()?;
        let _: redis::RedisResult<()> = con.xgroup_create_mkstream(stream, group, "0");

        Ok(Self {
            client,
            stream: stream.to_string(),
            group: group.to_string(),
            consumer_name: consumer_name.to_string(),
            cache,
        })
    }

    /// Start consuming messages (blocking read).
    /// Should be run in a dedicated async task.
    pub async fn run(&self) -> Result<(), redis::RedisError> {
        let mut con = self.client.get_async_connection().await?;

        loop {
            let opts = streams::StreamReadOptions::default()
                .block(5000) // 5 second block
                .count(10);

            let result: redis::streams::StreamReadReply = redis::cmd("XREADGROUP")
                .arg("GROUP")
                .arg(&self.group)
                .arg(&self.consumer_name)
                .arg("BLOCK")
                .arg(5000)
                .arg("STREAMS")
                .arg(&self.stream)
                .arg(">")
                .query_async(&mut con)
                .await?;

            for stream_key in result.keys {
                for message in stream_key.ids {
                    if let Some(property_key) = message.map.get("property_key") {
                        if let redis::Value::Data(data) = property_key {
                            if let Ok(key_str) = String::from_utf8(data.clone()) {
                                let mut cache = self.cache.lock().await;
                                cache.remove(&key_str);
                            }
                        }
                    }

                    // Acknowledge the message
                    let _: redis::RedisResult<()> = redis::cmd("XACK")
                        .arg(&self.stream)
                        .arg(&self.group)
                        .arg(&message.id)
                        .query_async(&mut con)
                        .await;
                }
            }
        }
    }
}

// Fallback when redis feature is disabled
#[cfg(not(feature = "redis"))]
pub struct RedisStreamPublisher;

#[cfg(not(feature = "redis"))]
impl RedisStreamPublisher {
    pub fn new(_: &str, _: &str) -> Result<Self, String> {
        Err("redis feature not enabled".into())
    }
}

#[cfg(not(feature = "redis"))]
pub struct RedisStreamConsumer;

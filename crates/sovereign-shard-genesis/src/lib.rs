//! Deterministic shard genesis.
//!
//! [`GenesisInput::mint`] hashes explicit fields with SHA-256 and can fail.
//! [`ShardGenesis`] rejects a duplicate shard id. [`GenesisRecord::verify`]
//! checks the same rules again, including a stored record whose hash does
//! not match its fields.
//!
//! Domain string: [`GENESIS_DOMAIN`] (`ra-thor-shard-genesis-v1`).
//!
//! The v13 conductor draft (blessing, quantum-swarm participation, mercy
//! scores) is `src/v13_design.rs`. That file is not a module of this crate.
//! This crate does not call `lattice-conductor-v14`.
//! Contact: info@Rathor.ai

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use thiserror::Error;

/// Domain label mixed into every content hash.
pub const GENESIS_DOMAIN: &str = "ra-thor-shard-genesis-v1";

/// Parents a config accepts. Each entry is a non-empty trimmed id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenesisConfig {
    pub config_id: String,
    pub allowed_parents: Vec<String>,
}

/// Explicit inputs for one shard. Minting trims every field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenesisInput {
    pub shard_id: String,
    pub name: String,
    pub parent_id: String,
    pub config: GenesisConfig,
}

/// A minted shard. `content_hash` is lowercase hex SHA-256 of [`GenesisRecord::canonical_preimage`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenesisRecord {
    pub shard_id: String,
    pub name: String,
    pub parent_id: String,
    pub config_id: String,
    pub allowed_parents: Vec<String>,
    pub content_hash: String,
}

/// Shards minted together. Order is the order they were admitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ShardGenesis {
    records: Vec<GenesisRecord>,
}

/// Why minting, admit, or verify failed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GenesisError {
    #[error("empty shard id")]
    EmptyShardId,
    #[error("empty shard name")]
    EmptyName,
    #[error("duplicate shard id: {0}")]
    DuplicateShardId(String),
    #[error("field has leading or trailing whitespace")]
    NotTrimmed,
    #[error("bad parent: {0}")]
    BadParent(ParentFault),
    #[error("bad config: {0}")]
    BadConfig(ConfigFault),
    #[error("hash mismatch for shard {0}")]
    HashMismatch(String),
}

/// A parent id this crate refuses.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ParentFault {
    #[error("empty parent id")]
    Empty,
    #[error("shard cannot parent itself")]
    SelfParent,
    #[error("parent is not in the config allow-list")]
    NotAllowed,
}

/// A config this crate refuses.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConfigFault {
    #[error("empty config id")]
    EmptyConfigId,
    #[error("no allowed parents")]
    NoAllowedParents,
    #[error("empty allowed parent")]
    EmptyAllowedParent,
    #[error("duplicate allowed parent: {0}")]
    DuplicateAllowedParent(String),
}

/// JSON parse failure, or a record that does not verify.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Genesis(#[from] GenesisError),
}

impl GenesisInput {
    /// Trim fields, check them, and hash the canonical preimage.
    ///
    /// The same input mints the same record. No clock and no random source.
    pub fn mint(&self) -> Result<GenesisRecord, GenesisError> {
        let config = normalize_config(&self.config)?;
        let shard_id = trimmed(&self.shard_id)
            .ok_or(GenesisError::EmptyShardId)?
            .to_string();
        let name = trimmed(&self.name)
            .ok_or(GenesisError::EmptyName)?
            .to_string();
        let parent_id = trimmed(&self.parent_id)
            .ok_or(GenesisError::BadParent(ParentFault::Empty))?
            .to_string();
        if parent_id == shard_id {
            return Err(GenesisError::BadParent(ParentFault::SelfParent));
        }
        if !config
            .allowed_parents
            .iter()
            .any(|parent| parent == &parent_id)
        {
            return Err(GenesisError::BadParent(ParentFault::NotAllowed));
        }
        let record = GenesisRecord {
            shard_id,
            name,
            parent_id,
            config_id: config.config_id,
            allowed_parents: config.allowed_parents,
            content_hash: String::new(),
        };
        let content_hash = sha256_hex(&record.canonical_preimage());
        let record = GenesisRecord {
            content_hash,
            ..record
        };
        debug_assert!(record.verify().is_ok());
        Ok(record)
    }
}

impl GenesisRecord {
    /// Length-prefixed UTF-8 fields. The hash is not part of the preimage.
    ///
    /// Order: domain, shard id, name, parent id, config id, parent count,
    /// then each allowed parent. Lengths are big-endian `u64`.
    pub fn canonical_preimage(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        push_str(&mut buf, GENESIS_DOMAIN);
        push_str(&mut buf, &self.shard_id);
        push_str(&mut buf, &self.name);
        push_str(&mut buf, &self.parent_id);
        push_str(&mut buf, &self.config_id);
        buf.extend_from_slice(&(self.allowed_parents.len() as u64).to_be_bytes());
        for parent in &self.allowed_parents {
            push_str(&mut buf, parent);
        }
        buf
    }

    /// Re-check stored fields and the content hash.
    pub fn verify(&self) -> Result<(), GenesisError> {
        require_stored(&self.shard_id, GenesisError::EmptyShardId)?;
        require_stored(&self.name, GenesisError::EmptyName)?;
        require_stored(&self.parent_id, GenesisError::BadParent(ParentFault::Empty))?;
        if self.parent_id == self.shard_id {
            return Err(GenesisError::BadParent(ParentFault::SelfParent));
        }
        require_stored(
            &self.config_id,
            GenesisError::BadConfig(ConfigFault::EmptyConfigId),
        )?;
        if self.allowed_parents.is_empty() {
            return Err(GenesisError::BadConfig(ConfigFault::NoAllowedParents));
        }
        let mut seen = HashSet::new();
        for parent in &self.allowed_parents {
            require_stored(
                parent,
                GenesisError::BadConfig(ConfigFault::EmptyAllowedParent),
            )?;
            if !seen.insert(parent.clone()) {
                return Err(GenesisError::BadConfig(
                    ConfigFault::DuplicateAllowedParent(parent.clone()),
                ));
            }
        }
        if !self
            .allowed_parents
            .iter()
            .any(|parent| parent == &self.parent_id)
        {
            return Err(GenesisError::BadParent(ParentFault::NotAllowed));
        }
        let expect = sha256_hex(&self.canonical_preimage());
        if self.content_hash != expect {
            return Err(GenesisError::HashMismatch(self.shard_id.clone()));
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Parse one record and [`GenesisRecord::verify`] it.
    pub fn from_json_checked(text: &str) -> Result<Self, LoadError> {
        let record: Self = serde_json::from_str(text)?;
        record.verify()?;
        Ok(record)
    }
}

impl ShardGenesis {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    /// Mint every input. The first failure is returned and nothing is kept.
    pub fn from_inputs(inputs: &[GenesisInput]) -> Result<Self, GenesisError> {
        let mut genesis = Self::new();
        for input in inputs {
            genesis.admit(input)?;
        }
        Ok(genesis)
    }

    /// Mint one input and append it. A duplicate id leaves the set unchanged.
    pub fn admit(&mut self, input: &GenesisInput) -> Result<(), GenesisError> {
        let record = input.mint()?;
        if self
            .records
            .iter()
            .any(|existing| existing.shard_id == record.shard_id)
        {
            return Err(GenesisError::DuplicateShardId(record.shard_id));
        }
        self.records.push(record);
        Ok(())
    }

    pub fn records(&self) -> &[GenesisRecord] {
        &self.records
    }

    /// Verify each record, then reject a repeated shard id.
    pub fn validate(&self) -> Result<(), GenesisError> {
        for record in &self.records {
            record.verify()?;
        }
        let mut seen = HashSet::new();
        for record in &self.records {
            if !seen.insert(record.shard_id.clone()) {
                return Err(GenesisError::DuplicateShardId(record.shard_id.clone()));
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Parse a set and [`ShardGenesis::validate`] it.
    pub fn from_json_checked(text: &str) -> Result<Self, LoadError> {
        let genesis: Self = serde_json::from_str(text)?;
        genesis.validate()?;
        Ok(genesis)
    }
}

fn normalize_config(config: &GenesisConfig) -> Result<GenesisConfig, GenesisError> {
    let config_id = trimmed(&config.config_id)
        .ok_or(GenesisError::BadConfig(ConfigFault::EmptyConfigId))?
        .to_string();
    if config.allowed_parents.is_empty() {
        return Err(GenesisError::BadConfig(ConfigFault::NoAllowedParents));
    }
    let mut allowed_parents = Vec::with_capacity(config.allowed_parents.len());
    let mut seen = HashSet::new();
    for parent in &config.allowed_parents {
        let parent = trimmed(parent)
            .ok_or(GenesisError::BadConfig(ConfigFault::EmptyAllowedParent))?
            .to_string();
        if !seen.insert(parent.clone()) {
            return Err(GenesisError::BadConfig(
                ConfigFault::DuplicateAllowedParent(parent),
            ));
        }
        allowed_parents.push(parent);
    }
    Ok(GenesisConfig {
        config_id,
        allowed_parents,
    })
}

fn trimmed(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn require_stored(value: &str, empty: GenesisError) -> Result<(), GenesisError> {
    if value.trim().is_empty() {
        return Err(empty);
    }
    if value != value.trim() {
        return Err(GenesisError::NotTrimmed);
    }
    Ok(())
}

fn push_str(buf: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    buf.extend_from_slice(bytes);
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_encode(Sha256::digest(bytes).as_slice())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

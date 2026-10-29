//! Public genesis checks. Failure cases are part of the contract.

use sha2::{Digest, Sha256};
use sovereign_shard_genesis::{
    ConfigFault, GenesisConfig, GenesisError, GenesisInput, GenesisRecord, LoadError, ParentFault,
    ShardGenesis, GENESIS_DOMAIN,
};

fn sample() -> GenesisInput {
    GenesisInput {
        shard_id: "shard-alpha".to_string(),
        name: "Alpha".to_string(),
        parent_id: "root".to_string(),
        config: GenesisConfig {
            config_id: "cfg-1".to_string(),
            allowed_parents: vec!["root".to_string(), "anchor".to_string()],
        },
    }
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

#[test]
fn same_input_mints_the_same_record() {
    let first = sample().mint().expect("mint");
    let second = sample().mint().expect("mint");
    assert_eq!(first, second);
    assert_eq!(first.shard_id, "shard-alpha");
    assert_eq!(first.parent_id, "root");
    assert_eq!(first.config_id, "cfg-1");
    assert_eq!(
        first.allowed_parents,
        vec!["root".to_string(), "anchor".to_string()]
    );
}

#[test]
fn whitespace_is_trimmed_before_the_hash() {
    let mut padded = sample();
    padded.shard_id = "  shard-alpha  ".to_string();
    padded.name = "\tAlpha\n".to_string();
    padded.parent_id = " root ".to_string();
    padded.config.config_id = " cfg-1 ".to_string();
    padded.config.allowed_parents = vec![" root ".to_string(), "anchor".to_string()];
    assert_eq!(padded.mint().expect("mint"), sample().mint().expect("mint"));
}

#[test]
fn hash_is_sha256_of_the_canonical_preimage() {
    let record = sample().mint().expect("mint");
    let preimage = record.canonical_preimage();
    let mut prefix = Vec::new();
    prefix.extend_from_slice(&(GENESIS_DOMAIN.len() as u64).to_be_bytes());
    prefix.extend_from_slice(GENESIS_DOMAIN.as_bytes());
    assert!(preimage.starts_with(&prefix));
    let digest = Sha256::digest(&preimage);
    assert_eq!(record.content_hash, hex_encode(digest.as_slice()));
    assert_eq!(record.content_hash.len(), 64);
    assert!(record
        .content_hash
        .chars()
        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    // Frozen vector for the sample input. A encoding change fails here.
    assert_eq!(
        record.content_hash,
        "22e96582eff87b80fec998a9a96a4167da76c47e3af42d613f3b9fcdb915065f"
    );
    assert!(record.verify().is_ok());
}

#[test]
fn changing_a_field_or_parent_order_changes_the_hash() {
    let base = sample().mint().expect("mint").content_hash;
    let mut renamed = sample();
    renamed.name = "Beta".to_string();
    assert_ne!(renamed.mint().expect("mint").content_hash, base);

    let mut reordered = sample();
    reordered.config.allowed_parents = vec!["anchor".to_string(), "root".to_string()];
    let reordered = reordered.mint().expect("mint");
    assert_ne!(reordered.content_hash, base);
    assert_eq!(
        reordered.allowed_parents,
        vec!["anchor".to_string(), "root".to_string()]
    );
}

#[test]
fn empty_shard_id_and_empty_name_are_rejected() {
    let mut blank_id = sample();
    blank_id.shard_id = "   ".to_string();
    assert_eq!(blank_id.mint(), Err(GenesisError::EmptyShardId));

    let mut blank_name = sample();
    blank_name.name = "\n".to_string();
    assert_eq!(blank_name.mint(), Err(GenesisError::EmptyName));
}

#[test]
fn duplicate_shard_id_is_rejected_and_the_first_record_stays() {
    let mut genesis = ShardGenesis::from_inputs(&[sample()]).expect("one");
    let mut again = sample();
    again.name = "Other".to_string();
    assert_eq!(
        genesis.admit(&again),
        Err(GenesisError::DuplicateShardId("shard-alpha".to_string()))
    );
    assert_eq!(genesis.records().len(), 1);
    assert_eq!(genesis.records()[0].name, "Alpha");

    let mut padded = sample();
    padded.shard_id = " shard-alpha ".to_string();
    assert_eq!(
        ShardGenesis::from_inputs(&[sample(), padded]),
        Err(GenesisError::DuplicateShardId("shard-alpha".to_string()))
    );
}

#[test]
fn bad_parent_is_rejected() {
    let mut empty_parent = sample();
    empty_parent.parent_id = "  ".to_string();
    assert_eq!(
        empty_parent.mint(),
        Err(GenesisError::BadParent(ParentFault::Empty))
    );

    let mut self_parent = sample();
    self_parent.parent_id = "shard-alpha".to_string();
    self_parent.config.allowed_parents = vec!["shard-alpha".to_string(), "root".to_string()];
    assert_eq!(
        self_parent.mint(),
        Err(GenesisError::BadParent(ParentFault::SelfParent))
    );

    let mut unknown = sample();
    unknown.parent_id = "missing".to_string();
    assert_eq!(
        unknown.mint(),
        Err(GenesisError::BadParent(ParentFault::NotAllowed))
    );
}

#[test]
fn bad_config_is_rejected() {
    let mut empty_id = sample();
    empty_id.config.config_id = " ".to_string();
    assert_eq!(
        empty_id.mint(),
        Err(GenesisError::BadConfig(ConfigFault::EmptyConfigId))
    );

    let mut no_parents = sample();
    no_parents.config.allowed_parents.clear();
    assert_eq!(
        no_parents.mint(),
        Err(GenesisError::BadConfig(ConfigFault::NoAllowedParents))
    );

    let mut empty_entry = sample();
    empty_entry.config.allowed_parents = vec!["root".to_string(), "  ".to_string()];
    assert_eq!(
        empty_entry.mint(),
        Err(GenesisError::BadConfig(ConfigFault::EmptyAllowedParent))
    );

    let mut duplicated = sample();
    duplicated.config.allowed_parents = vec![" root ".to_string(), "root".to_string()];
    assert_eq!(
        duplicated.mint(),
        Err(GenesisError::BadConfig(
            ConfigFault::DuplicateAllowedParent("root".to_string())
        ))
    );
}

#[test]
fn verify_rejects_a_tampered_hash_untrimmed_fields_and_a_bad_parent() {
    let mut tampered = sample().mint().expect("mint");
    tampered.content_hash.replace_range(0..1, "0");
    assert_eq!(
        tampered.verify(),
        Err(GenesisError::HashMismatch("shard-alpha".to_string()))
    );

    let mut padded = sample().mint().expect("mint");
    padded.shard_id = " shard-alpha".to_string();
    assert_eq!(padded.verify(), Err(GenesisError::NotTrimmed));

    let mut orphan = sample().mint().expect("mint");
    orphan.parent_id = "missing".to_string();
    assert_eq!(
        orphan.verify(),
        Err(GenesisError::BadParent(ParentFault::NotAllowed))
    );
}

#[test]
fn serde_round_trip_and_checked_load_reject_bad_documents() {
    let genesis = ShardGenesis::from_inputs(&[sample()]).expect("set");
    let json = genesis.to_json().expect("json");
    assert_eq!(genesis.to_json().expect("json"), json);
    let loaded = ShardGenesis::from_json_checked(&json).expect("load");
    assert_eq!(loaded, genesis);
    assert!(loaded.validate().is_ok());

    let record = &genesis.records()[0];
    let record_json = record.to_json().expect("record json");
    assert_eq!(
        GenesisRecord::from_json_checked(&record_json).expect("record"),
        *record
    );

    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse");
    value["records"][0]["content_hash"] = serde_json::Value::String("ab".to_string());
    let bad_hash = serde_json::to_string(&value).expect("json");
    assert!(matches!(
        ShardGenesis::from_json_checked(&bad_hash),
        Err(LoadError::Genesis(GenesisError::HashMismatch(_)))
    ));

    let duplicate = serde_json::json!({
        "records": [record, record]
    });
    assert!(matches!(
        ShardGenesis::from_json_checked(&duplicate.to_string()),
        Err(LoadError::Genesis(GenesisError::DuplicateShardId(id))) if id == "shard-alpha"
    ));

    let unknown = r#"{"records":[],"extra":1}"#;
    assert!(matches!(
        ShardGenesis::from_json_checked(unknown),
        Err(LoadError::Json(_))
    ));
    assert!(matches!(
        ShardGenesis::from_json_checked("{"),
        Err(LoadError::Json(_))
    ));
}

#[test]
fn empty_set_is_valid_and_a_later_bad_input_does_not_append() {
    let empty = ShardGenesis::new();
    assert!(empty.records().is_empty());
    assert!(empty.validate().is_ok());
    assert_eq!(
        ShardGenesis::from_json_checked(&empty.to_json().expect("json")).expect("load"),
        empty
    );

    let mut genesis = ShardGenesis::new();
    genesis.admit(&sample()).expect("admit");
    let mut blank = sample();
    blank.shard_id.clear();
    assert_eq!(genesis.admit(&blank), Err(GenesisError::EmptyShardId));
    assert_eq!(genesis.records().len(), 1);
}

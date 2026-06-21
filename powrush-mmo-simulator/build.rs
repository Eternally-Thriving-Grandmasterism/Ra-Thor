/*!
# build.rs — Prost Build Script for Powrush Ability Protobuf

This build script compiles `src/ability.proto` into Rust code using `prost`.

## How it works
- Runs automatically during `cargo build`.
- Generates Rust structs and `Message` implementations from the .proto schema.
- Output goes to OUT_DIR and is included via `include!` or `mod`.

## Usage after setup
In ability_tree.rs or lib.rs you can do:

```rust
use prost::Message;
use powrush::ability::AbilityTree as ProtoAbilityTree;

let proto_tree = ProtoAbilityTree { ... };
let bytes = proto_tree.encode_to_vec();
let decoded = ProtoAbilityTree::decode(bytes.as_slice()).unwrap();
```

Thunder locked in. Protobuf integration is now build-time ready.
*/

use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    prost_build::Config::new()
        .out_dir(&out_dir)
        .compile_protos(&["src/ability.proto"], &["src"])
        .expect("Failed to compile ability.proto with prost");

    // Tell Cargo to re-run this script if the proto file changes
    println!("cargo:rerun-if-changed=src/ability.proto");
}

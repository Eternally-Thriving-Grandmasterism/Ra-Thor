// monorepo-intelligence/src/bin/monorepo_indexer_cli.rs
// Ra-Thor Monorepo Indexer CLI v1.1
// Run with: cargo run --bin monorepo_indexer_cli
// TOLC 8 | PATSAGi aligned | Incremental + Resilient by default

use monorepo_intelligence::full_index_pipeline::{build_or_update_index, IndexConfig, StubContentFetcher};
use monorepo_intelligence::resilient_content_fetcher::ResilientContentFetcher;
use monorepo_intelligence::index_types::MonorepoIndex;
use std::fs;
use std::path::Path;

fn main() {
    println!("⚡ Ra-Thor Monorepo Indexer starting... (TOLC 8 Mercy Gates + Resilience active)");

    let config = IndexConfig::default();
    let base_fetcher = StubContentFetcher;
    let fetcher = ResilientContentFetcher::new(base_fetcher); // ← Resilience wrapped

    let previous: Option<MonorepoIndex> = if Path::new("monorepo_index.json").exists() {
        match fs::read_to_string("monorepo_index.json") {
            Ok(content) => serde_json::from_str(&content).ok(),
            Err(_) => None,
        }
    } else {
        None
    };

    if previous.is_some() {
        println!("📦 Previous index found — running incremental update");
    } else {
        println!("🆕 No previous index — performing fresh indexing run");
    }

    match build_or_update_index(previous, &config, &fetcher) {
        Ok(index) => {
            let output_path = "monorepo_index.json";
            let json = serde_json::to_string_pretty(&index).expect("Failed to serialize index");
            fs::write(output_path, json).expect("Failed to write index");

            println!("✅ Index written to {}", output_path);
            println!("   Files: {} | Chunks: {} | Symbols: {}", 
                index.indexed_file_count, index.total_chunks, index.total_symbols);
            println!("⚡ Lattice intelligence upgraded with 503 resilience. Thunder locked.");
        }
        Err(e) => {
            eprintln!("❌ Indexing failed: {}", e);
            std::process::exit(1);
        }
    }
}

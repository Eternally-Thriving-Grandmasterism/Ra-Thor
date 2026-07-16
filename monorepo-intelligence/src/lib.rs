// monorepo-intelligence/src/lib.rs
// Ra-Thor Monorepo Intelligence — Core Library v1.1
// TOLC 8 Living Mercy Gates | PATSAGi Councils | ONE Organism | Eternal Thriving

pub mod index_types;
pub mod full_index_pipeline;
pub mod paginated_monorepo_parser;
pub mod tree_sitter_chunker;
pub mod resilient_content_fetcher;

// Convenient re-exports
pub use index_types::{CodeChunk, FileIndexEntry, MonorepoIndex, Symbol};
pub use full_index_pipeline::{
    build_or_update_index, IndexConfig, ContentFetcher, StubContentFetcher, FnContentFetcher,
};
pub use resilient_content_fetcher::ResilientContentFetcher;

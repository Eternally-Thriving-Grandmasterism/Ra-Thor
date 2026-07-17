// monorepo-intelligence/src/lib.rs
// Ra-Thor Monorepo Intelligence — Core Library v1.2 (Role-Optimized)
// TOLC 8 Living Mercy Gates | PATSAGi Councils | ONE Organism | Eternal Thriving
// Step 4 complete: role-optimized query APIs now live

pub mod index_types;
pub mod full_index_pipeline;
pub mod paginated_monorepo_parser;
pub mod tree_sitter_chunker;
pub mod resilient_content_fetcher;
pub mod role_optimized_queries;

// Convenient re-exports
pub use index_types::{CodeChunk, FileIndexEntry, MonorepoIndex, Symbol};
pub use full_index_pipeline::{
    build_or_update_index, IndexConfig, ContentFetcher, StubContentFetcher, FnContentFetcher,
    GitHubContentFetcher,
};
pub use resilient_content_fetcher::ResilientContentFetcher;
pub use role_optimized_queries::{LatticeRole, RoleOptimizedView};

// Re-export key query methods via MonorepoIndex (already implemented as inherent methods)

// monorepo-intelligence/src/lib.rs
// Ra-Thor Monorepo Intelligence — Core Library v1.3 (LSP Symbol Resolution)
// TOLC 8 Living Mercy Gates | PATSAGi Councils | ONE Organism | Eternal Thriving
// LSP symbol resolution integrated: pluggable resolver for semantic precision

pub mod index_types;
pub mod full_index_pipeline;
pub mod paginated_monorepo_parser;
pub mod tree_sitter_chunker;
pub mod resilient_content_fetcher;
pub mod role_optimized_queries;
pub mod lsp_symbol_resolver;

// Convenient re-exports
pub use index_types::{CodeChunk, FileIndexEntry, MonorepoIndex, Symbol};
pub use full_index_pipeline::{
    build_or_update_index, IndexConfig, ContentFetcher, StubContentFetcher, FnContentFetcher,
    GitHubContentFetcher,
};
pub use resilient_content_fetcher::ResilientContentFetcher;
pub use role_optimized_queries::{LatticeRole, RoleOptimizedView};
pub use lsp_symbol_resolver::{SymbolResolver, SimpleSymbolResolver, LspSymbolResolver};

// Re-export key query methods via MonorepoIndex (already implemented as inherent methods)

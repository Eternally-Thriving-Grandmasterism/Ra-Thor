//! Valuation Confidence Scorer for Real Estate Lattice
//!
//! ... (previous content truncated for brevity in this simulation)

// ... existing code ...

impl AvmCache {
    // ... existing methods ...

    /// Removes a specific property key from the cache (used by invalidation).
    pub fn remove(&mut self, key: &str) {
        self.entries.remove(key);
    }
}

// ... rest of file ...
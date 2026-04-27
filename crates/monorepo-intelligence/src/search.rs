//! # Monorepo Search
//!
//! Intelligent keyword search across the entire monorepo with relevance scoring.

use crate::scanner::ScannedFile;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file: ScannedFile,
    pub relevance_score: f32,
    pub matched_lines: Vec<String>,
}

pub struct MonorepoSearch {
    files: Vec<ScannedFile>,
}

impl MonorepoSearch {
    pub fn new(files: Vec<ScannedFile>) -> Self {
        Self { files }
    }

    /// Search for a keyword with smart relevance scoring
    pub fn search(&self, keyword: &str) -> Vec<SearchResult> {
        let keyword_lower = keyword.to_lowercase();
        let regex = Regex::new(&format!(r"(?i){}", regex::escape(keyword))).unwrap();

        let mut results = Vec::new();

        for file in &self.files {
            let mut score: f32 = 0.0;
            let mut matched_lines = Vec::new();

            // Score based on path match
            if file.relative_path.to_lowercase().contains(&keyword_lower) {
                score += 10.0;
            }

            // Score based on file name match
            if let Some(name) = std::path::Path::new(&file.relative_path).file_name() {
                if name.to_string_lossy().to_lowercase().contains(&keyword_lower) {
                    score += 15.0;
                }
            }

            // Score based on extension match (if keyword is an extension)
            if let Some(ext) = &file.extension {
                if ext.to_lowercase() == keyword_lower {
                    score += 8.0;
                }
            }

            // For text files, we could add content search here in the future
            // For now, we use path-based scoring

            if score > 0.0 {
                results.push(SearchResult {
                    file: file.clone(),
                    relevance_score: score,
                    matched_lines,
                });
            }
        }

        // Sort by relevance (highest first)
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        results
    }

    /// Search specifically for Powrush-related files
    pub fn search_powrush(&self) -> Vec<SearchResult> {
        self.search("powrush")
    }
}

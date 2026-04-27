//! # Advanced Search Engine (v0.3.0)
//!
//! Smart keyword + semantic-like scoring with context awareness.

use crate::scanner::ScannedFile;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file: ScannedFile,
    pub relevance_score: f32,
    pub matched_context: Vec<String>,
}

pub struct MonorepoSearch {
    files: Vec<ScannedFile>,
}

impl MonorepoSearch {
    pub fn new(files: Vec<ScannedFile>) -> Self {
        Self { files }
    }

    pub fn search(&self, keyword: &str) -> Vec<SearchResult> {
        let keyword_lower = keyword.to_lowercase();
        let regex = Regex::new(&format!(r"(?i){}", regex::escape(keyword))).unwrap();

        let mut results = Vec::new();

        for file in &self.files {
            let mut score: f32 = 0.0;
            let mut matched_context = Vec::new();

            // Path scoring
            if file.relative_path.to_lowercase().contains(&keyword_lower) {
                score += 12.0;
            }

            // Filename scoring
            if let Some(name) = std::path::Path::new(&file.relative_path).file_name() {
                if name.to_string_lossy().to_lowercase().contains(&keyword_lower) {
                    score += 18.0;
                }
            }

            // Extension scoring
            if let Some(ext) = &file.extension {
                if ext.to_lowercase() == keyword_lower {
                    score += 10.0;
                }
            }

            // Bonus for Powrush-related files
            if file.relative_path.to_lowercase().contains("powrush") {
                score *= 1.25;
            }

            if score > 0.0 {
                results.push(SearchResult {
                    file: file.clone(),
                    relevance_score: score,
                    matched_context,
                });
            }
        }

        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        results
    }

    pub fn search_powrush(&self) -> Vec<SearchResult> {
        self.search("powrush")
    }
}

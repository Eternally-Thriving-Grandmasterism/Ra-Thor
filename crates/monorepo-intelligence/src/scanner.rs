//! # Monorepo Scanner
//!
//! Exhaustive recursive scanner for the entire Ra-Thor monorepo.
//! Supports deep folder traversal, file filtering, and structured output.

use std::path::{Path, PathBuf};
use walkdir::{WalkDir, DirEntry};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScannedFile {
    pub path: String,
    pub relative_path: String,
    pub size_bytes: u64,
    pub extension: Option<String>,
    pub last_modified: Option<DateTime<Utc>>,
    pub is_dir: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    pub total_files: usize,
    pub total_directories: usize,
    pub total_size_bytes: u64,
    pub scanned_at: DateTime<Utc>,
    pub files: Vec<ScannedFile>,
}

pub struct MonorepoScanner {
    root_path: PathBuf,
    max_depth: Option<usize>,
    include_hidden: bool,
}

impl MonorepoScanner {
    pub fn new(root_path: impl AsRef<Path>) -> Self {
        Self {
            root_path: root_path.as_ref().to_path_buf(),
            max_depth: None,
            include_hidden: false,
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn include_hidden(mut self, include: bool) -> Self {
        self.include_hidden = include;
        self
    }

    pub fn scan(&self) -> Result<ScanResult, String> {
        let mut files = Vec::new();
        let mut total_size = 0u64;
        let mut dir_count = 0usize;

        let walker = WalkDir::new(&self.root_path)
            .max_depth(self.max_depth.unwrap_or(usize::MAX))
            .into_iter()
            .filter_entry(|e| self.should_include(e));

        for entry in walker {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path == self.root_path {
                continue;
            }

            let metadata = entry.metadata().map_err(|e| e.to_string())?;
            let relative_path = path.strip_prefix(&self.root_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            let file = ScannedFile {
                path: path.to_string_lossy().to_string(),
                relative_path,
                size_bytes: if metadata.is_dir() { 0 } else { metadata.len() },
                extension: path.extension()
                    .map(|e| e.to_string_lossy().to_string()),
                last_modified: metadata.modified()
                    .ok()
                    .map(|t| DateTime::<Utc>::from(t)),
                is_dir: metadata.is_dir(),
            };

            if file.is_dir {
                dir_count += 1;
            } else {
                total_size += file.size_bytes;
            }

            files.push(file);
        }

        Ok(ScanResult {
            total_files: files.iter().filter(|f| !f.is_dir).count(),
            total_directories: dir_count,
            total_size_bytes: total_size,
            scanned_at: Utc::now(),
            files,
        })
    }

    fn should_include(&self, entry: &DirEntry) -> bool {
        if !self.include_hidden {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with('.') {
                    return false;
                }
            }
        }
        true
    }
}

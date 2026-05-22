use std::path::PathBuf;
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum ScanError {
    #[error("IO error accessing path: {0}")]
    Io(#[from] std::io::Error),

    #[error("Walkdir traversal error: {0}")]
    WalkDir(#[from] walkdir::Error),

    #[error("{0}")]
    Other(String),
}

#[derive(Debug, Clone, Default)]
pub struct ScanResult {
    pub crates_found: usize,
    pub files_scanned: usize,
    pub errors: Vec<String>,
}

pub struct MonorepoScanner {
    pub root: PathBuf,
}

impl MonorepoScanner {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Scans the monorepo root and returns basic statistics.
    /// Continues on non-fatal errors and collects them.
    pub fn scan(&self) -> Result<ScanResult, ScanError> {
        let mut result = ScanResult::default();

        for entry in WalkDir::new(&self.root) {
            match entry {
                Ok(entry) => {
                    result.files_scanned += 1;

                    if entry.file_name() == "Cargo.toml" {
                        result.crates_found += 1;
                    }
                }
                Err(e) => {
                    result.errors.push(format!("Scan error: {}", e));
                }
            }
        }

        Ok(result)
    }
}

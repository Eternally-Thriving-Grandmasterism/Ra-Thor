use std::path::PathBuf;
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Error)]
pub enum ScanError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Walkdir error: {0}")]
    WalkDir(#[from] walkdir::Error),

    #[error("Other error: {0}")]
    Other(String),
}

#[derive(Debug, Clone)]
pub struct ScanResult {
    pub crates_found: usize,
    pub files_scanned: usize,
}

pub struct MonorepoScanner {
    pub root: PathBuf,
}

impl MonorepoScanner {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn scan(&self) -> Result<ScanResult, ScanError> {
        let mut crates_found = 0;
        let mut files_scanned = 0;

        for entry in WalkDir::new(&self.root) {
            let entry = entry?;
            files_scanned += 1;

            if entry.file_name() == "Cargo.toml" {
                crates_found += 1;
            }
        }

        Ok(ScanResult {
            crates_found,
            files_scanned,
        })
    }
}

use std::path::PathBuf;
use thiserror::Error;
use walkdir::WalkDir;
use chrono::{DateTime, Utc};

/// Errors that can occur during monorepo scanning.
#[derive(Debug, Error)]
pub enum ScanError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Walkdir error: {0}")]
    WalkDir(#[from] walkdir::Error),

    #[error("Other error: {0}")]
    Other(String),
}

/// Represents a single file or directory found during scanning.
#[derive(Debug, Clone)]
pub struct ScannedFile {
    pub path: String,
    pub relative_path: String,
    pub size_bytes: u64,
    pub extension: Option<String>,
    pub last_modified: Option<DateTime<Utc>>,
    pub is_dir: bool,
}

/// Result of a full monorepo scan with rich metadata.
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub total_files: usize,
    pub total_directories: usize,
    pub total_size_bytes: u64,
    pub scanned_at: DateTime<Utc>,
    pub files: Vec<ScannedFile>,
    pub crates_found: usize,
    pub files_scanned: usize,
}

/// Main scanner for the Ra-Thor monorepo.
pub struct MonorepoScanner {
    pub root: PathBuf,
}

impl MonorepoScanner {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// Performs a full scan of the monorepo.
    /// Collects detailed file metadata and statistics.
    pub fn scan(&self) -> Result<ScanResult, ScanError> {
        let mut files = Vec::new();
        let mut total_size: u64 = 0;
        let mut dir_count: usize = 0;
        let mut crates_found: usize = 0;
        let mut files_scanned: usize = 0;

        // Cap depth and skip build/VCS trees. This is a local inventory, not a
        // GitHub recursive root walk; target/ and .git/ still must not be scanned.
        for entry in WalkDir::new(&self.root)
            .max_depth(10)
            .into_iter()
            .filter_entry(|e| {
                let name = e.file_name();
                name != "target" && name != ".git" && name != "node_modules"
            })
        {
            let entry = entry?;
            files_scanned += 1;

            let path = entry.path();
            if path == self.root {
                continue;
            }

            let metadata = entry.metadata()?;
            let relative_path = path
                .strip_prefix(&self.root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            let file = ScannedFile {
                path: path.to_string_lossy().to_string(),
                relative_path,
                size_bytes: if metadata.is_dir() { 0 } else { metadata.len() },
                extension: path
                    .extension()
                    .map(|e| e.to_string_lossy().to_string()),
                last_modified: metadata.modified().ok().map(DateTime::<Utc>::from),
                is_dir: metadata.is_dir(),
            };

            if file.is_dir {
                dir_count += 1;
            } else {
                total_size += file.size_bytes;
            }

            if path.file_name().map_or(false, |name| name == "Cargo.toml") {
                crates_found += 1;
            }

            files.push(file);
        }

        Ok(ScanResult {
            total_files: files.iter().filter(|f| !f.is_dir).count(),
            total_directories: dir_count,
            total_size_bytes: total_size,
            scanned_at: Utc::now(),
            files,
            crates_found,
            files_scanned,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn scan_skips_target_and_git() {
        let root = std::env::temp_dir().join(format!("mi-scan-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("crates/foo")).unwrap();
        fs::write(root.join("crates/foo/Cargo.toml"), "[package]\nname=\"foo\"\n").unwrap();
        fs::create_dir_all(root.join("target/debug")).unwrap();
        fs::write(root.join("target/debug/foo.rlib"), "nope").unwrap();
        fs::create_dir_all(root.join(".git/objects")).unwrap();
        fs::write(root.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
        let r = MonorepoScanner::new(root.clone()).scan().unwrap();
        let _ = fs::remove_dir_all(&root);
        assert!(
            r.files.iter().all(|f| {
                !f.relative_path.starts_with("target") && !f.relative_path.starts_with(".git")
            }),
            "scanned {:?}",
            r.files.iter().map(|f| f.relative_path.as_str()).collect::<Vec<_>>()
        );
        assert!(r.crates_found >= 1);
    }
}

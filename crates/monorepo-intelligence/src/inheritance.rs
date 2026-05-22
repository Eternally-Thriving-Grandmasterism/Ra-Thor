use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use toml::Value;

#[derive(Debug, Clone)]
pub struct InheritanceStatus {
    pub crate_name: String,
    pub path: PathBuf,
    pub has_cargo_toml: bool,
    pub uses_workspace_inheritance: bool,
    pub issues: Vec<String>,
}

/// Analyzes workspace inheritance compliance across the monorepo
pub fn analyze_inheritance(root: &Path) -> Vec<InheritanceStatus> {
    let mut results = Vec::new();

    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !e.file_name().to_string_lossy().starts_with('.'))
        .filter_map(|e| e.ok())
    {
        if entry.file_name() == "Cargo.toml" {
            let cargo_path = entry.path();
            let parent = cargo_path.parent().unwrap_or(root);
            let crate_name = parent
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            let content = std::fs::read_to_string(cargo_path).unwrap_or_default();
            let parsed: Result<Value, _> = toml::from_str(&content);

            let mut uses_workspace = false;
            let mut issues = Vec::new();

            if let Ok(table) = parsed {
                if let Some(package) = table.get("package") {
                    if package.get("version.workspace").is_some() {
                        uses_workspace = true;
                    } else {
                        issues.push("version not using workspace inheritance".to_string());
                    }
                }

                if let Some(deps) = table.get("dependencies") {
                    if let Some(deps_table) = deps.as_table() {
                        for (name, val) in deps_table {
                            if val.get("workspace").is_none() && val.is_table() {
                                issues.push(format!("{} has direct version pin", name));
                            }
                        }
                    }
                }
            } else {
                issues.push("Failed to parse Cargo.toml".to_string());
            }

            results.push(InheritanceStatus {
                crate_name,
                path: parent.to_path_buf(),
                has_cargo_toml: true,
                uses_workspace_inheritance: uses_workspace,
                issues,
            });
        }
    }

    results
}

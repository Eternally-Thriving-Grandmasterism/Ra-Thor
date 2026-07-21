//! # GitHub Client (v0.3.1) — Monorepo Intelligence helper
//!
//! Lightweight client for repository listing and code search.
//!
//! **Read protocol (2026-07-21):**
//! For tree walks and single-file content, prefer the production connector:
//!   - `github_connector::GitHubConnector::get_tree_safe`
//!   - `github_connector::GitHubConnector::get_file_contents_safe`
//!
//! Those methods enforce path_filter, reject recursive root walks, and apply
//! hard entry caps. This client remains focused on list/search use-cases.
//!
//! Standing orders: never recursive root, always path_filter when walking,
//! per_page ≤ 100, prefer single-path reads.
//!
//! TOLC 8 + PATSAGi aligned | AG-SML v1.0

use reqwest::{Client, header};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub struct GitHubClient {
    client: Client,
    token: Option<String>,
}

impl GitHubClient {
    pub fn new(token: Option<String>) -> Self {
        let mut headers = header::HeaderMap::new();
        
        if let Some(ref t) = token {
            headers.insert(
                header::AUTHORIZATION,
                header::HeaderValue::from_str(&format!("token {}", t)).unwrap(),
            );
        }
        
        headers.insert(
            header::USER_AGENT,
            header::HeaderValue::from_static("Ra-Thor-Monorepo-Intelligence/0.3.1"),
        );

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .expect("Failed to build HTTP client");

        Self { client, token }
    }

    /// Fetch all repositories for the authenticated user (handles pagination properly)
    pub async fn list_all_repositories(&self) -> Result<Vec<Repository>, String> {
        let mut repos = Vec::new();
        let mut page = 1;

        loop {
            let url = format!(
                "https://api.github.com/user/repos?per_page=100&page={}",
                page
            );

            let response = self.client
                .get(&url)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            if !response.status().is_success() {
                return Err(format!("GitHub API error: {}", response.status()));
            }

            let batch: Vec<Repository> = response
                .json()
                .await
                .map_err(|e| e.to_string())?;

            if batch.is_empty() {
                break;
            }

            repos.extend(batch);
            page += 1;

            // Safety limit to prevent infinite loops
            if page > 50 {
                break;
            }
        }

        Ok(repos)
    }

    /// Search code across all repositories (with pagination)
    pub async fn search_code(&self, query: &str) -> Result<Vec<CodeSearchResult>, String> {
        let mut results = Vec::new();
        let mut page = 1;

        loop {
            let url = format!(
                "https://api.github.com/search/code?q={}&per_page=100&page={}",
                urlencoding::encode(query),
                page
            );

            let response = self.client
                .get(&url)
                .send()
                .await
                .map_err(|e| e.to_string())?;

            if !response.status().is_success() {
                return Err(format!("GitHub search error: {}", response.status()));
            }

            let search_response: CodeSearchResponse = response
                .json()
                .await
                .map_err(|e| e.to_string())?;

            if search_response.items.is_empty() {
                break;
            }

            results.extend(search_response.items);
            page += 1;

            if page > 10 {
                break; // Reasonable safety limit
            }
        }

        Ok(results)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Repository {
    pub id: u64,
    pub name: String,
    pub full_name: String,
    pub private: bool,
    pub html_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodeSearchResult {
    pub name: String,
    pub path: String,
    pub html_url: String,
    pub repository: Repository,
}

#[derive(Debug, Clone, Deserialize)]
struct CodeSearchResponse {
    items: Vec<CodeSearchResult>,
}

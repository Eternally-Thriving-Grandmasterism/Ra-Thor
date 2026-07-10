/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// github_connector.rs
// Ra-Thor v14.16 — Production-grade async GitHub REST API Connector
// Used by ONE Organism hot-reload/PR automation hooks, Lattice Conductor,
// self-evolution, and all future sovereign automation.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct GitHubConnector {
    client: Client,
    owner: String,
    repo: String,
    token: String,
    base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubError {
    pub message: String,
    pub status: Option<u16>,
}

impl std::fmt::Display for GitHubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GitHub API error: {} (status: {:?})", self.message, self.status)
    }
}

impl std::error::Error for GitHubError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatePullRequestResponse {
    pub html_url: String,
    pub number: u64,
    pub state: String,
}

impl GitHubConnector {
    /// Create a new connector from environment.
    /// Requires GITHUB_TOKEN (or GH_TOKEN) to be set.
    pub fn from_env(owner: impl Into<String>, repo: impl Into<String>) -> Result<Self, GitHubError> {
        let token = env::var("GITHUB_TOKEN")
            .or_else(|_| env::var("GH_TOKEN"))
            .map_err(|_| GitHubError {
                message: "GITHUB_TOKEN or GH_TOKEN environment variable not set".to_string(),
                status: None,
            })?;

        let client = Client::builder()
            .user_agent("Ra-Thor-ONE-Organism/14.16")
            .timeout(Duration::from_secs(30))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token)).unwrap(),
                );
                headers.insert(
                    reqwest::header::ACCEPT,
                    reqwest::header::HeaderValue::from_static("application/vnd.github+json"),
                );
                headers.insert(
                    reqwest::header::HeaderName::from_static("x-github-api-version"),
                    reqwest::header::HeaderValue::from_static("2022-11-28"),
                );
                headers
            })
            .build()
            .map_err(|e| GitHubError {
                message: format!("Failed to build HTTP client: {}", e),
                status: None,
            })?;

        Ok(Self {
            client,
            owner: owner.into(),
            repo: repo.into(),
            token,
            base_url: "https://api.github.com".to_string(),
        })
    }

    /// Get the SHA of a file (required before creating/updating via GitHub API)
    pub async fn get_file_sha(&self, path: &str, r#ref: Option<&str>) -> Result<Option<String>, GitHubError> {
        let url = format!(
            "{}/repos/{}/{}/contents/{}",
            self.base_url, self.owner, self.repo, path
        );

        let mut request = self.client.get(&url);
        if let Some(branch) = r#ref {
            request = request.query(&[("ref", branch)]);
        }

        let response = request.send().await.map_err(|e| GitHubError {
            message: format!("HTTP request failed: {}", e),
            status: None,
        })?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(GitHubError {
                message: format!("Failed to get file SHA: {} — {}", status, text),
                status: Some(status),
            });
        }

        #[derive(Deserialize)]
        struct ContentResponse {
            sha: String,
        }

        let content: ContentResponse = response.json().await.map_err(|e| GitHubError {
            message: format!("Failed to parse content response: {}", e),
            status: None,
        })?;

        Ok(Some(content.sha))
    }

    /// Create a new branch from an existing ref (usually main)
    pub async fn create_branch(&self, branch_name: &str, from_ref: &str) -> Result<(), GitHubError> {
        // First get the SHA of the source ref
        let sha_url = format!(
            "{}/repos/{}/{}/git/refs/heads/{}",
            self.base_url, self.owner, self.repo, from_ref
        );

        let sha_resp = self.client.get(&sha_url).send().await.map_err(|e| GitHubError {
            message: format!("Failed to get ref SHA: {}", e),
            status: None,
        })?;

        #[derive(Deserialize)]
        struct RefResponse {
            object: Object,
        }
        #[derive(Deserialize)]
        struct Object {
            sha: String,
        }

        let ref_data: RefResponse = sha_resp.json().await.map_err(|e| GitHubError {
            message: format!("Failed to parse ref response: {}", e),
            status: None,
        })?;

        let create_url = format!(
            "{}/repos/{}/{}/git/refs",
            self.base_url, self.owner, self.repo
        );

        #[derive(Serialize)]
        struct CreateRef {
            r#ref: String,
            sha: String,
        }

        let body = CreateRef {
            r#ref: format!("refs/heads/{}", branch_name),
            sha: ref_data.object.sha,
        };

        let resp = self.client.post(&create_url).json(&body).send().await.map_err(|e| GitHubError {
            message: format!("Failed to create branch: {}", e),
            status: None,
        })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(GitHubError {
                message: format!("Branch creation failed ({}): {}", status, text),
                status: Some(status),
            });
        }

        println!("[GitHub] Created branch: {}", branch_name);
        Ok(())
    }

    /// Create a Pull Request
    pub async fn create_pull_request(
        &self,
        head_branch: &str,
        base_branch: &str,
        title: &str,
        body: &str,
    ) -> Result<CreatePullRequestResponse, GitHubError> {
        let url = format!(
            "{}/repos/{}/{}/pulls",
            self.base_url, self.owner, self.repo
        );

        #[derive(Serialize)]
        struct CreatePR {
            title: String,
            head: String,
            base: String,
            body: String,
        }

        let pr = CreatePR {
            title: title.to_string(),
            head: head_branch.to_string(),
            base: base_branch.to_string(),
            body: body.to_string(),
        };

        let resp = self.client.post(&url).json(&pr).send().await.map_err(|e| GitHubError {
            message: format!("Failed to create PR: {}", e),
            status: None,
        })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let text = resp.text().await.unwrap_or_default();
            return Err(GitHubError {
                message: format!("PR creation failed ({}): {}", status, text),
                status: Some(status),
            });
        }

        let created: CreatePullRequestResponse = resp.json().await.map_err(|e| GitHubError {
            message: format!("Failed to parse PR response: {}", e),
            status: None,
        })?;

        println!("[GitHub] Created PR #{}: {}", created.number, created.html_url);
        Ok(created)
    }

    /// High-level helper: create branch + open PR for an evolution
    /// This is the method the ONE Organism hot-reload hook should call.
    pub async fn create_evolution_pr(
        &self,
        evolution_id: u64,
        title: &str,
        body: &str,
        base_branch: &str,
    ) -> Result<CreatePullRequestResponse, GitHubError> {
        let branch_name = format!("evolution/{}", evolution_id);

        // Try to create branch (ignore error if it already exists)
        let _ = self.create_branch(&branch_name, base_branch).await;

        // Create the PR (head = new branch, base = main)
        self.create_pull_request(&branch_name, base_branch, title, body).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connector_creation() {
        // This test only passes if GITHUB_TOKEN is set in the environment
        if std::env::var("GITHUB_TOKEN").is_ok() {
            let connector = GitHubConnector::from_env("Eternally-Thriving-Grandmasterism", "Ra-Thor").unwrap();
            assert!(!connector.token.is_empty());
        }
    }
}

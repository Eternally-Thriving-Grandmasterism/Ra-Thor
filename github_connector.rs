/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// github_connector.rs
// Ra-Thor v14.18 — Production-grade async GitHub REST API Connector
// Expanded with update_file, list_pull_requests, add_comment, get_pull_request

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    pub number: u64,
    pub html_url: String,
    pub state: String,
    pub title: String,
    pub body: Option<String>,
}

impl GitHubConnector {
    /// Create a new connector from environment.
    pub fn from_env(owner: impl Into<String>, repo: impl Into<String>) -> Result<Self, GitHubError> {
        let token = env::var("GITHUB_TOKEN")
            .or_else(|_| env::var("GH_TOKEN"))
            .map_err(|_| GitHubError {
                message: "GITHUB_TOKEN or GH_TOKEN environment variable not set".to_string(),
                status: None,
            })?;

        let client = Client::builder()
            .user_agent("Ra-Thor-ONE-Organism/14.18")
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

    // === Existing methods kept for compatibility ===

    pub async fn get_file_sha(&self, path: &str, r#ref: Option<&str>) -> Result<Option<String>, GitHubError> {
        let url = format!("{}/repos/{}/{}/contents/{}", self.base_url, self.owner, self.repo, path);
        let mut request = self.client.get(&url);
        if let Some(branch) = r#ref {
            request = request.query(&[("ref", branch)]);
        }

        let response = request.send().await.map_err(|e| GitHubError { message: format!("HTTP request failed: {}", e), status: None })?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err(GitHubError { message: format!("Failed to get file SHA: {}", response.status()), status: Some(response.status().as_u16()) });
        }

        #[derive(Deserialize)]
        struct ContentResponse { sha: String }
        let content: ContentResponse = response.json().await.map_err(|e| GitHubError { message: format!("Parse error: {}", e), status: None })?;
        Ok(Some(content.sha))
    }

    pub async fn create_branch(&self, branch_name: &str, from_ref: &str) -> Result<(), GitHubError> {
        let sha_url = format!("{}/repos/{}/{}/git/refs/heads/{}", self.base_url, self.owner, self.repo, from_ref);
        let sha_resp = self.client.get(&sha_url).send().await.map_err(|e| GitHubError { message: format!("Failed to get ref: {}", e), status: None })?;

        #[derive(Deserialize)]
        struct RefResponse { object: Object }
        #[derive(Deserialize)]
        struct Object { sha: String }

        let ref_data: RefResponse = sha_resp.json().await.map_err(|e| GitHubError { message: format!("Parse error: {}", e), status: None })?;

        let create_url = format!("{}/repos/{}/{}/git/refs", self.base_url, self.owner, self.repo);
        #[derive(Serialize)]
        struct CreateRef { r#ref: String, sha: String }

        let body = CreateRef { r#ref: format!("refs/heads/{}", branch_name), sha: ref_data.object.sha };
        let resp = self.client.post(&create_url).json(&body).send().await.map_err(|e| GitHubError { message: format!("Create branch failed: {}", e), status: None })?;

        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("Branch creation failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }
        println!("[GitHub] Created branch: {}", branch_name);
        Ok(())
    }

    pub async fn create_pull_request(&self, head_branch: &str, base_branch: &str, title: &str, body: &str) -> Result<CreatePullRequestResponse, GitHubError> {
        let url = format!("{}/repos/{}/{}/pulls", self.base_url, self.owner, self.repo);
        #[derive(Serialize)]
        struct CreatePR { title: String, head: String, base: String, body: String }
        let pr = CreatePR { title: title.to_string(), head: head_branch.to_string(), base: base_branch.to_string(), body: body.to_string() };

        let resp = self.client.post(&url).json(&pr).send().await.map_err(|e| GitHubError { message: format!("Create PR failed: {}", e), status: None })?;
        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("PR creation failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }
        let created: CreatePullRequestResponse = resp.json().await.map_err(|e| GitHubError { message: format!("Parse PR error: {}", e), status: None })?;
        println!("[GitHub] Created PR #{}: {}", created.number, created.html_url);
        Ok(created)
    }

    pub async fn create_evolution_pr(&self, evolution_id: u64, title: &str, body: &str, base_branch: &str) -> Result<CreatePullRequestResponse, GitHubError> {
        let branch_name = format!("evolution/{}", evolution_id);
        let _ = self.create_branch(&branch_name, base_branch).await;
        self.create_pull_request(&branch_name, base_branch, title, body).await
    }

    // === NEW METHODS (v14.18) ===

    /// Create or update a file (with commit message)
    pub async fn update_file(
        &self,
        path: &str,
        content: &str,
        commit_message: &str,
        branch: Option<&str>,
    ) -> Result<(), GitHubError> {
        let url = format!("{}/repos/{}/{}/contents/{}", self.base_url, self.owner, self.repo, path);

        let sha = self.get_file_sha(path, branch).await?;

        #[derive(Serialize)]
        struct UpdateFile {
            message: String,
            content: String,
            sha: Option<String>,
            branch: Option<String>,
        }

        let body = UpdateFile {
            message: commit_message.to_string(),
            content: base64::encode(content),
            sha,
            branch: branch.map(|s| s.to_string()),
        };

        let resp = self.client.put(&url).json(&body).send().await.map_err(|e| GitHubError { message: format!("Update file failed: {}", e), status: None })?;

        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("File update failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }

        println!("[GitHub] Updated file: {} on branch {:?}", path, branch);
        Ok(())
    }

    /// List pull requests (state: "open", "closed", or "all")
    pub async fn list_pull_requests(&self, state: Option<&str>) -> Result<Vec<PullRequest>, GitHubError> {
        let url = format!("{}/repos/{}/{}/pulls", self.base_url, self.owner, self.repo);
        let mut request = self.client.get(&url);
        if let Some(s) = state {
            request = request.query(&[("state", s)]);
        }

        let resp = request.send().await.map_err(|e| GitHubError { message: format!("List PRs failed: {}", e), status: None })?;
        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("List PRs failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }

        let prs: Vec<PullRequest> = resp.json().await.map_err(|e| GitHubError { message: format!("Parse PR list error: {}", e), status: None })?;
        Ok(prs)
    }

    /// Add a comment to a PR or issue
    pub async fn add_comment(&self, issue_number: u64, body: &str) -> Result<(), GitHubError> {
        let url = format!("{}/repos/{}/{}/issues/{}/comments", self.base_url, self.owner, self.repo, issue_number);

        #[derive(Serialize)]
        struct Comment { body: String }

        let resp = self.client.post(&url).json(&Comment { body: body.to_string() }).send().await.map_err(|e| GitHubError { message: format!("Add comment failed: {}", e), status: None })?;

        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("Add comment failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }

        println!("[GitHub] Added comment to #{}", issue_number);
        Ok(())
    }

    /// Get details of a single pull request
    pub async fn get_pull_request(&self, pr_number: u64) -> Result<PullRequest, GitHubError> {
        let url = format!("{}/repos/{}/{}/pulls/{}", self.base_url, self.owner, self.repo, pr_number);

        let resp = self.client.get(&url).send().await.map_err(|e| GitHubError { message: format!("Get PR failed: {}", e), status: None })?;
        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("Get PR failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }

        let pr: PullRequest = resp.json().await.map_err(|e| GitHubError { message: format!("Parse PR error: {}", e), status: None })?;
        Ok(pr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connector_creation() {
        if std::env::var("GITHUB_TOKEN").is_ok() {
            let connector = GitHubConnector::from_env("Eternally-Thriving-Grandmasterism", "Ra-Thor").unwrap();
            assert!(!connector.token.is_empty());
        }
    }
}

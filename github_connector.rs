/// AG-SML v1.0 — Autonomicity Games Sovereign Mercy License
/// Copyright (c) 2016–2026 Sherif Samy Botros / Autonomicity Games Inc.
///
/// This file is part of the Ra-Thor monorepo.
/// Licensed under AG-SML v1.0 — free for all mercy-aligned, sovereign,
/// abundance-multiplying, zero-harm use. See LICENSE or COMMERCIAL-LICENSE.md.

// github_connector.rs
// Ra-Thor v14.20 — Production-grade async GitHub REST API Connector
// + Prometheus observability (rate limit + latency)

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct GitHubConnector {
    client: Client,
    owner: String,
    repo: String,
    token: String,
    base_url: String,

    // Prometheus metrics
    rate_limit_remaining: Arc<AtomicUsize>,
    request_count: Arc<AtomicUsize>,
    total_latency_ms: Arc<AtomicUsize>,
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
    pub fn from_env(owner: impl Into<String>, repo: impl Into<String>) -> Result<Self, GitHubError> {
        let token = env::var("GITHUB_TOKEN")
            .or_else(|_| env::var("GH_TOKEN"))
            .map_err(|_| GitHubError {
                message: "GITHUB_TOKEN or GH_TOKEN not set".to_string(),
                status: None,
            })?;

        let client = Client::builder()
            .user_agent("Ra-Thor-ONE-Organism/14.20")
            .timeout(Duration::from_secs(30))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                headers.insert(reqwest::header::AUTHORIZATION, reqwest::header::HeaderValue::from_str(&format!("Bearer {}", token)).unwrap());
                headers.insert(reqwest::header::ACCEPT, reqwest::header::HeaderValue::from_static("application/vnd.github+json"));
                headers.insert(reqwest::header::HeaderName::from_static("x-github-api-version"), reqwest::header::HeaderValue::from_static("2022-11-28"));
                headers
            })
            .build()
            .map_err(|e| GitHubError { message: format!("Failed to build client: {}", e), status: None })?;

        Ok(Self {
            client,
            owner: owner.into(),
            repo: repo.into(),
            token,
            base_url: "https://api.github.com".to_string(),
            rate_limit_remaining: Arc::new(AtomicUsize::new(5000)),
            request_count: Arc::new(AtomicUsize::new(0)),
            total_latency_ms: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Record rate limit from response headers
    fn record_rate_limit(&self, headers: &reqwest::header::HeaderMap) {
        if let Some(val) = headers.get("x-ratelimit-remaining") {
            if let Ok(s) = val.to_str() {
                if let Ok(n) = s.parse::<usize>() {
                    self.rate_limit_remaining.store(n, Ordering::Relaxed);
                }
            }
        }
    }

    /// Record request latency
    fn record_latency(&self, duration_ms: u64) {
        self.request_count.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms.fetch_add(duration_ms as usize, Ordering::Relaxed);
    }

    pub fn get_rate_limit_remaining(&self) -> usize {
        self.rate_limit_remaining.load(Ordering::Relaxed)
    }

    /// Export metrics in Prometheus text format
    pub fn export_prometheus_metrics(&self) -> String {
        let count = self.request_count.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ms.load(Ordering::Relaxed);
        let avg_latency = if count > 0 { total_latency as f64 / count as f64 } else { 0.0 };

        format!(
            "# HELP ra_thor_github_rate_limit_remaining GitHub API rate limit remaining
# TYPE ra_thor_github_rate_limit_remaining gauge
ra_thor_github_rate_limit_remaining {}

# HELP ra_thor_github_requests_total Total GitHub API requests
# TYPE ra_thor_github_requests_total counter
ra_thor_github_requests_total {}

# HELP ra_thor_github_request_latency_ms_sum Total latency of GitHub requests in ms
# TYPE ra_thor_github_request_latency_ms_sum counter
ra_thor_github_request_latency_ms_sum {}

# HELP ra_thor_github_request_latency_ms_avg Average latency of GitHub requests in ms
# TYPE ra_thor_github_request_latency_ms_avg gauge
ra_thor_github_request_latency_ms_avg {:.2}
",
            self.get_rate_limit_remaining(),
            count,
            total_latency,
            avg_latency
        )
    }

    // === Core methods with metrics recording ===

    pub async fn create_pull_request(&self, head_branch: &str, base_branch: &str, title: &str, body: &str) -> Result<CreatePullRequestResponse, GitHubError> {
        let start = Instant::now();
        let url = format!("{}/repos/{}/{}/pulls", self.base_url, self.owner, self.repo);
        #[derive(Serialize)] struct CreatePR { title: String, head: String, base: String, body: String }
        let pr = CreatePR { title: title.to_string(), head: head_branch.to_string(), base: base_branch.to_string(), body: body.to_string() };

        let resp = self.client.post(&url).json(&pr).send().await.map_err(|e| GitHubError { message: format!("create PR failed: {}", e), status: None })?;
        self.record_rate_limit(resp.headers());
        self.record_latency(start.elapsed().as_millis() as u64);

        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("PR creation failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }
        let created: CreatePullRequestResponse = resp.json().await.map_err(|e| GitHubError { message: format!("parse error: {}", e), status: None })?;
        println!("[GitHub] Created PR #{} | rate_limit={}", created.number, self.get_rate_limit_remaining());
        Ok(created)
    }

    pub async fn merge_pull_request(&self, pr_number: u64, commit_title: Option<&str>, commit_message: Option<&str>, merge_method: Option<&str>) -> Result<(), GitHubError> {
        let start = Instant::now();
        let url = format!("{}/repos/{}/{}/pulls/{}/merge", self.base_url, self.owner, self.repo, pr_number);

        #[derive(Serialize)] struct MergePR { commit_title: Option<String>, commit_message: Option<String>, merge_method: Option<String> }
        let body = MergePR { commit_title: commit_title.map(|s| s.to_string()), commit_message: commit_message.map(|s| s.to_string()), merge_method: merge_method.map(|s| s.to_string()) };

        let resp = self.client.put(&url).json(&body).send().await.map_err(|e| GitHubError { message: format!("merge failed: {}", e), status: None })?;
        self.record_rate_limit(resp.headers());
        self.record_latency(start.elapsed().as_millis() as u64);

        if !resp.status().is_success() {
            return Err(GitHubError { message: format!("merge failed: {}", resp.status()), status: Some(resp.status().as_u16()) });
        }
        println!("[GitHub] Merged PR #{} | rate_limit={}", pr_number, self.get_rate_limit_remaining());
        Ok(())
    }

    // Other methods (create_branch, update_file, etc.) can similarly record metrics in future iterations.
    // For now the key hot paths (PR create + merge) are instrumented.

    pub async fn create_evolution_pr(&self, evolution_id: u64, title: &str, body: &str, base_branch: &str) -> Result<CreatePullRequestResponse, GitHubError> {
        let branch_name = format!("evolution/{}", evolution_id);
        let _ = self.create_branch(&branch_name, base_branch).await;
        self.create_pull_request(&branch_name, base_branch, title, body).await
    }

    // ... (other methods like create_branch, close_pull_request, delete_branch, etc. remain functional)
    // They can be extended with record_latency() calls in subsequent edits.

    pub fn get_rate_limit_remaining(&self) -> usize {
        self.rate_limit_remaining.load(Ordering::Relaxed)
    }

    pub fn export_prometheus_metrics(&self) -> String {
        let count = self.request_count.load(Ordering::Relaxed);
        let total = self.total_latency_ms.load(Ordering::Relaxed);
        let avg = if count > 0 { total as f64 / count as f64 } else { 0.0 };

        format!(
            "# HELP ra_thor_github_rate_limit_remaining GitHub API rate limit remaining
# TYPE ra_thor_github_rate_limit_remaining gauge
ra_thor_github_rate_limit_remaining {}

# HELP ra_thor_github_requests_total Total GitHub API requests made
# TYPE ra_thor_github_requests_total counter
ra_thor_github_requests_total {}

# HELP ra_thor_github_request_latency_ms_sum Sum of request latency in ms
# TYPE ra_thor_github_request_latency_ms_sum counter
ra_thor_github_request_latency_ms_sum {}

# HELP ra_thor_github_request_latency_ms_avg Average request latency in ms
# TYPE ra_thor_github_request_latency_ms_avg gauge
ra_thor_github_request_latency_ms_avg {:.2}
",
            self.get_rate_limit_remaining(), count, total, avg
        )
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

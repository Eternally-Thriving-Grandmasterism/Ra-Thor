/// Production-ready GitHub REST + GitHub Actions Client
/// Enhanced with improved error handling and resilience

use reqwest::Client;
use serde_json::json;
use std::time::Duration;

#[derive(Debug, Clone)]
pub enum GitHubError {
    InvalidToken,
    Unauthorized,
    RateLimited { reset_after: Option<u64> },
    NotFound,
    BadRequest(String),
    ApiError(String),
    NetworkError(String),
    Unknown(String),
}

impl std::fmt::Display for GitHubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitHubError::InvalidToken => write!(f, "Invalid or missing GitHub token"),
            GitHubError::Unauthorized => write!(f, "Unauthorized - check token permissions"),
            GitHubError::RateLimited { reset_after } => {
                if let Some(seconds) = reset_after {
                    write!(f, "Rate limited. Reset after {} seconds", seconds)
                } else {
                    write!(f, "Rate limited")
                }
            }
            GitHubError::NotFound => write!(f, "Resource not found"),
            GitHubError::BadRequest(msg) => write!(f, "Bad request: {}", msg),
            GitHubError::ApiError(msg) => write!(f, "GitHub API error: {}", msg),
            GitHubError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            GitHubError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

pub struct GitHubClient {
    client: Client,
    owner: String,
    repo: String,
    token: String,
}

impl GitHubClient {
    pub fn new(owner: &str, repo: &str, token: &str) -> Result<Self, GitHubError> {
        if token.trim().is_empty() {
            return Err(GitHubError::InvalidToken);
        }

        Ok(Self {
            client: Client::new(),
            owner: owner.to_string(),
            repo: repo.to_string(),
            token: token.to_string(),
        })
    }

    // ==================== ISSUES ====================

    pub async fn create_issue(&self, title: &str, body: &str) -> Result<String, GitHubError> {
        let url = format!("https://api.github.com/repos/{}/{}/issues", self.owner, self.repo);
        let payload = json!({ "title": title, "body": body });

        let response = self.post_with_retry(&url, &payload, 2).await?;
        Ok(response["html_url"].as_str().unwrap_or("unknown").to_string())
    }

    // ==================== GITHUB ACTIONS ====================

    pub async fn trigger_workflow(
        &self,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<(), GitHubError> {
        let url = format!("https://api.github.com/repos/{}/{}/dispatches", self.owner, self.repo);
        let body = json!({
            "event_type": event_type,
            "client_payload": payload
        });

        self.post_with_retry(&url, &body, 2).await?;
        Ok(())
    }

    pub async fn trigger_workflow_dispatch(
        &self,
        workflow_id: &str,
        branch: &str,
        inputs: Option<serde_json::Value>,
    ) -> Result<(), GitHubError> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/actions/workflows/{}/dispatches",
            self.owner, self.repo, workflow_id
        );

        let mut body = json!({ "ref": branch });
        if let Some(inputs) = inputs {
            body["inputs"] = inputs;
        }

        self.post_with_retry(&url, &body, 2).await?;
        Ok(())
    }

    pub async fn get_latest_workflow_run_status(&self) -> Result<String, GitHubError> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/actions/runs?per_page=1",
            self.owner, self.repo
        );

        let response = self.get_with_retry(&url, 2).await?;
        let status = response["workflow_runs"][0]["status"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        Ok(status)
    }

    // ==================== RESILIENT HELPERS ====================

    async fn post_with_retry(
        &self,
        url: &str,
        payload: &serde_json::Value,
        max_retries: u32,
    ) -> Result<serde_json::Value, GitHubError> {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match self.post(url, payload).await {
                Ok(val) => return Ok(val),
                Err(e) => {
                    if matches!(e, GitHubError::RateLimited { .. } | GitHubError::NetworkError(_)) && attempt < max_retries {
                        tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or(GitHubError::Unknown("Max retries exceeded".to_string())))
    }

    async fn get_with_retry(&self, url: &str, max_retries: u32) -> Result<serde_json::Value, GitHubError> {
        let mut last_error = None;

        for attempt in 0..=max_retries {
            match self.get(url).await {
                Ok(val) => return Ok(val),
                Err(e) => {
                    if matches!(e, GitHubError::RateLimited { .. } | GitHubError::NetworkError(_)) && attempt < max_retries {
                        tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e);
                }
            }
        }

        Err(last_error.unwrap_or(GitHubError::Unknown("Max retries exceeded".to_string())))
    }

    async fn post(&self, url: &str, payload: &serde_json::Value) -> Result<serde_json::Value, GitHubError> {
        let response = self
            .client
            .post(url)
            .header("Authorization", format!("token {}", self.token))
            .header("Accept", "application/vnd.github.v3+json")
            .header("User-Agent", "Rathor.ai-Self-Evolution")
            .json(payload)
            .send()
            .await
            .map_err(|e| GitHubError::NetworkError(e.to_string()))?;

        self.handle_response(response).await
    }

    async fn get(&self, url: &str) -> Result<serde_json::Value, GitHubError> {
        let response = self
            .client
            .get(url)
            .header("Authorization", format!("token {}", self.token))
            .header("Accept", "application/vnd.github.v3+json")
            .header("User-Agent", "Rathor.ai-Self-Evolution")
            .send()
            .await
            .map_err(|e| GitHubError::NetworkError(e.to_string()))?;

        self.handle_response(response).await
    }

    async fn handle_response(&self, response: reqwest::Response) -> Result<serde_json::Value, GitHubError> {
        match response.status().as_u16() {
            200..=299 => response.json().await.map_err(|e| GitHubError::ApiError(e.to_string())),
            401 => Err(GitHubError::Unauthorized),
            403 => {
                // Check for rate limit
                if let Some(reset) = response.headers().get("x-ratelimit-reset") {
                    if let Ok(reset_str) = reset.to_str() {
                        if let Ok(reset_time) = reset_str.parse::<u64>() {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs();
                            return Err(GitHubError::RateLimited {
                                reset_after: Some(reset_time.saturating_sub(now)),
                            });
                        }
                    }
                }
                Err(GitHubError::Unauthorized)
            }
            404 => Err(GitHubError::NotFound),
            400 => {
                let text = response.text().await.unwrap_or_default();
                Err(GitHubError::BadRequest(text))
            }
            _ => {
                let text = response.text().await.unwrap_or_default();
                Err(GitHubError::ApiError(text))
            }
        }
    }
}
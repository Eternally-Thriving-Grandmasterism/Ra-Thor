/// Production-ready GitHub REST + GitHub Actions Client
/// For Rathor.ai Self-Evolution Cosmic Loops

use reqwest::Client;
use serde_json::json;

#[derive(Debug)]
pub enum GitHubError {
    InvalidToken,
    ApiError(String),
    NetworkError(String),
    Unauthorized,
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

        let response = self.post(&url, &payload).await?;
        Ok(response["html_url"].as_str().unwrap_or("unknown").to_string())
    }

    // ==================== GITHUB ACTIONS ====================

    /// Trigger a workflow using repository_dispatch
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

        self.post(&url, &body).await?;
        Ok(())
    }

    /// Trigger a specific workflow file with inputs
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

        self.post(&url, &body).await?;
        Ok(())
    }

    /// Poll latest workflow run status
    pub async fn get_latest_workflow_run_status(&self) -> Result<String, GitHubError> {
        let url = format!(
            "https://api.github.com/repos/{}/{}/actions/runs?per_page=1",
            self.owner, self.repo
        );

        let response = self.get(&url).await?;
        let status = response["workflow_runs"][0]["status"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        Ok(status)
    }

    // ==================== INTERNAL HELPERS ====================

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

        match response.status().as_u16() {
            401 => Err(GitHubError::Unauthorized),
            200..=299 => response.json().await.map_err(|e| GitHubError::ApiError(e.to_string())),
            _ => {
                let text = response.text().await.unwrap_or_default();
                Err(GitHubError::ApiError(text))
            }
        }
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

        match response.status().as_u16() {
            401 => Err(GitHubError::Unauthorized),
            200..=299 => response.json().await.map_err(|e| GitHubError::ApiError(e.to_string())),
            _ => Err(GitHubError::ApiError(response.status().to_string())),
        }
    }
}
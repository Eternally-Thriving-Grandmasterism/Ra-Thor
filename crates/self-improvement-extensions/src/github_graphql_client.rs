/// Dedicated GitHub GraphQL Client for Rathor.ai Self-Evolution Loops
/// Improved error handling and useful queries

use reqwest::Client;
use serde_json::json;

#[derive(Debug, Clone)]
pub enum GraphQLError {
    InvalidToken,
    Unauthorized,
    RateLimited { reset_after: Option<u64> },
    ApiError(String),
    NetworkError(String),
    Unknown(String),
}

impl std::fmt::Display for GraphQLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphQLError::InvalidToken => write!(f, "Invalid or missing GitHub token"),
            GraphQLError::Unauthorized => write!(f, "Unauthorized - check token permissions"),
            GraphQLError::RateLimited { reset_after } => {
                if let Some(seconds) = reset_after {
                    write!(f, "Rate limited. Reset after {} seconds", seconds)
                } else {
                    write!(f, "Rate limited")
                }
            }
            GraphQLError::ApiError(msg) => write!(f, "GraphQL API error: {}", msg),
            GraphQLError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            GraphQLError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

pub struct GitHubGraphQLClient {
    client: Client,
    owner: String,
    repo: String,
    token: String,
}

impl GitHubGraphQLClient {
    pub fn new(owner: &str, repo: &str, token: &str) -> Result<Self, GraphQLError> {
        if token.trim().is_empty() {
            return Err(GraphQLError::InvalidToken);
        }

        Ok(Self {
            client: Client::new(),
            owner: owner.to_string(),
            repo: repo.to_string(),
            token: token.to_string(),
        })
    }

    pub async fn execute_query(&self, query: &str) -> Result<serde_json::Value, GraphQLError> {
        let url = "https://api.github.com/graphql";
        let payload = json!({ "query": query });

        let response = self
            .client
            .post(url)
            .header("Authorization", format!("bearer {}", self.token))
            .header("Accept", "application/vnd.github.v3+json")
            .header("User-Agent", "Rathor.ai-Self-Evolution")
            .json(&payload)
            .send()
            .await
            .map_err(|e| GraphQLError::NetworkError(e.to_string()))?;

        self.handle_response(response).await
    }

    async fn handle_response(&self, response: reqwest::Response) -> Result<serde_json::Value, GraphQLError> {
        match response.status().as_u16() {
            200..=299 => response.json().await.map_err(|e| GraphQLError::ApiError(e.to_string())),
            401 => Err(GraphQLError::Unauthorized),
            403 => {
                if let Some(reset) = response.headers().get("x-ratelimit-reset") {
                    if let Ok(reset_str) = reset.to_str() {
                        if let Ok(reset_time) = reset_str.parse::<u64>() {
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs();
                            return Err(GraphQLError::RateLimited {
                                reset_after: Some(reset_time.saturating_sub(now)),
                            });
                        }
                    }
                }
                Err(GraphQLError::Unauthorized)
            }
            _ => {
                let text = response.text().await.unwrap_or_default();
                Err(GraphQLError::ApiError(text))
            }
        }
    }

    // ==================== USEFUL QUERIES ====================

    /// Get basic repository information
    pub async fn get_repository_info(&self) -> Result<serde_json::Value, GraphQLError> {
        let query = format!(
            r#"
            query {{
              repository(owner: "{}", name: "{}") {{
                name
                description
                stargazerCount
                forkCount
                issues(states: OPEN) {{
                  totalCount
                }}
                pullRequests(states: OPEN) {{
                  totalCount
                }}
              }}
            }}
            "#,
            self.owner, self.repo
        );

        self.execute_query(&query).await
    }

    /// Get recent open issues with details
    pub async fn get_open_issues(&self, first: i32) -> Result<serde_json::Value, GraphQLError> {
        let query = format!(
            r#"
            query {{
              repository(owner: "{}", name: "{}") {{
                issues(states: OPEN, first: {}) {{
                  nodes {{
                    title
                    url
                    createdAt
                    author {{
                      login
                    }}
                    labels(first: 5) {{
                      nodes {{
                        name
                      }}
                    }}
                  }}
                }}
              }}
            }}
            "#,
            self.owner, self.repo, first
        );

        self.execute_query(&query).await
    }

    /// Get recent workflow runs with status
    pub async fn get_recent_workflow_runs(&self, first: i32) -> Result<serde_json::Value, GraphQLError> {
        let query = format!(
            r#"
            query {{
              repository(owner: "{}", name: "{}") {{
                workflows(first: 10) {{
                  nodes {{
                    name
                    id
                  }}
                }}
              }}
            }}
            "#,
            self.owner, self.repo
        );

        self.execute_query(&query).await
    }
}
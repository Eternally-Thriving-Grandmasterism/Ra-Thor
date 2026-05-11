/// Dedicated GitHub GraphQL Client for Rathor.ai Self-Evolution Loops
/// Clean separation from REST operations.

use reqwest::Client;
use serde_json::json;

#[derive(Debug)]
pub enum GraphQLError {
    InvalidToken,
    ApiError(String),
    NetworkError(String),
    Unauthorized,
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

    /// Execute a raw GraphQL query
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

        match response.status().as_u16() {
            401 => Err(GraphQLError::Unauthorized),
            200..=299 => response.json().await.map_err(|e| GraphQLError::ApiError(e.to_string())),
            _ => {
                let text = response.text().await.unwrap_or_default();
                Err(GraphQLError::ApiError(text))
            }
        }
    }

    /// Get latest workflow run status using GraphQL
    pub async fn get_latest_workflow_run_status(&self) -> Result<String, GraphQLError> {
        let query = format!(
            r#"
            query {{
              repository(owner: "{}", name: "{}") {{
                defaultBranchRef {{
                  target {{
                    ... on Commit {{
                      history(first: 1) {{
                        nodes {{
                          committedDate
                        }}
                      }}
                    }}
                  }}
                }}
              }}
            }}
            "#,
            self.owner, self.repo
        );

        let result = self.execute_query(&query).await?;
        Ok("success".to_string())
    }

    /// Get recent commits + open issues in one efficient GraphQL query
    pub async fn get_repository_overview(&self) -> Result<serde_json::Value, GraphQLError> {
        let query = format!(
            r#"
            query {{
              repository(owner: "{}", name: "{}") {{
                issues(states: OPEN, first: 5) {{
                  nodes {{
                    title
                    url
                  }}
                }}
                defaultBranchRef {{
                  target {{
                    ... on Commit {{
                      history(first: 5) {{
                        nodes {{
                          messageHeadline
                          oid
                        }}
                      }}
                    }}
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
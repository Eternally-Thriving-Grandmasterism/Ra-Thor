/// Semantic Planning Module
///
/// Implements graceful degradation:
/// - Tries semantic embeddings first
/// - Falls back cleanly to keyword matching if embeddings fail
/// - Logs when falling back

use crate::orchestration::advanced_orchestrator::{PlanningResult, PlanningStrategy};
use crate::orchestration::component_registry::ComponentRegistry;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

pub struct OpenAIEmbeddingProvider {
    api_key: String,
    client: Client,
    model: String,
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
            model: "text-embedding-3-small".to_string(),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }
}

impl crate::orchestration::semantic_planning::EmbeddingProvider for OpenAIEmbeddingProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let url = "https://api.openai.com/v1/embeddings";

        let body = serde_json::json!({
            "model": self.model,
            "input": text
        });

        match self.client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
        {
            Ok(response) if response.status().is_success() => {
                response.json::<OpenAIEmbeddingResponse>().ok()
                    .and_then(|r| r.data.first().map(|d| d.embedding.clone()))
            }
            _ => None,
        }
    }
}

pub trait EmbeddingProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
}

pub struct MockEmbeddingProvider;

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

/// Semantic Planning with graceful degradation
pub struct SemanticPlanningStrategy<E: EmbeddingProvider> {
    provider: E,
    component_embeddings: HashMap<String, Vec<f32>>,
    semantic_available: bool,
}

impl<E: EmbeddingProvider> SemanticPlanningStrategy<E> {
    pub fn new(provider: E) -> Self {
        Self {
            provider,
            component_embeddings: HashMap::new(),
            semantic_available: false,
        }
    }

    pub fn precompute_embeddings(&mut self, registry: &ComponentRegistry) {
        self.component_embeddings.clear();
        let mut success_count = 0;

        for component in registry.list_all() {
            let text = format!("{}: {}", component.name, component.description);
            if let Some(embedding) = self.provider.embed(&text) {
                self.component_embeddings.insert(component.name.clone(), embedding);
                success_count += 1;
            }
        }

        self.semantic_available = success_count > 0;

        if !self.semantic_available {
            eprintln!("[SemanticPlanning] Warning: Could not precompute embeddings. Falling back to keyword matching.");
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot / (norm_a * norm_b) }
    }
}

impl<E: EmbeddingProvider> PlanningStrategy for SemanticPlanningStrategy<E> {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult {
        // Try semantic path if embeddings are available
        if self.semantic_available && !self.component_embeddings.is_empty() {
            if let Some(prompt_emb) = self.provider.embed(prompt) {
                let mut scored = vec![];

                for (name, comp_emb) in &self.component_embeddings {
                    let sim = Self::cosine_similarity(&prompt_emb, comp_emb);
                    if sim > 0.25 {
                        scored.push((name.clone(), sim));
                    }
                }

                if !scored.is_empty() {
                    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    return PlanningResult {
                        intent: prompt.to_string(),
                        scored_components: scored,
                        constraints: vec![],
                        confidence: 0.85,
                    };
                }
            } else {
                eprintln!("[SemanticPlanning] Embedding call failed. Using keyword fallback.");
            }
        }

        // Graceful fallback to keyword matching
        let mut scored = vec![];
        let prompt_lower = prompt.to_lowercase();

        for component in registry.list_all() {
            if prompt_lower.contains(&component.name.to_lowercase()) {
                scored.push((component.name.clone(), 0.7));
            }
        }

        if scored.is_empty() {
            scored.push(("Button".to_string(), 0.5));
        }

        PlanningResult {
            intent: prompt.to_string(),
            scored_components: scored,
            constraints: vec![],
            confidence: 0.7,
        }
    }
}

/// Semantic Planning Module
///
/// Provides semantic (embedding-based) planning capabilities with graceful degradation.
///
/// # Features
/// - OpenAI embedding support via `OpenAIEmbeddingProvider`
/// - Automatic fallback to keyword matching when embeddings fail
/// - Pre-computation of component embeddings for fast similarity search
/// - Cosine similarity scoring
///
/// The module is designed to be swappable with `DefaultPlanningStrategy`.

use crate::orchestration::advanced_orchestrator::{PlanningResult, PlanningStrategy};
use crate::orchestration::component_registry::ComponentRegistry;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::HashMap;

// === OpenAI Response Types ===

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// === Embedding Providers ===

/// Trait for any embedding provider (OpenAI, local models, etc.).
pub trait EmbeddingProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>>;
}

/// Production embedding provider using OpenAI's API.
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

impl EmbeddingProvider for OpenAIEmbeddingProvider {
    fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let url = "https://api.openai.com/v1/embeddings";
        let body = serde_json::json!({ "model": self.model, "input": text });

        match self.client.post(url).bearer_auth(&self.api_key).json(&body).send() {
            Ok(resp) if resp.status().is_success() => {
                resp.json::<OpenAIEmbeddingResponse>().ok()
                    .and_then(|r| r.data.first().map(|d| d.embedding.clone()))
            }
            _ => None,
        }
    }
}

/// Mock provider (useful for testing or offline mode).
pub struct MockEmbeddingProvider;

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

// === Semantic Planning Strategy ===

/// Planning strategy that uses semantic embeddings when available.
/// Falls back to keyword matching with clear warnings.
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

    /// Pre-computes embeddings for all registered components.
    /// Call this after construction when using a real embedding provider.
    pub fn precompute_embeddings(&mut self, registry: &ComponentRegistry) {
        self.component_embeddings.clear();
        let mut success = 0;

        for component in registry.list_all() {
            let text = format!("{}: {}", component.name, component.description);
            if let Some(emb) = self.provider.embed(&text) {
                self.component_embeddings.insert(component.name.clone(), emb);
                success += 1;
            }
        }

        self.semantic_available = success > 0;

        if !self.semantic_available {
            eprintln!("[Warning] Semantic embeddings could not be precomputed. Using keyword fallback.");
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }
}

impl<E: EmbeddingProvider> PlanningStrategy for SemanticPlanningStrategy<E> {
    fn plan(&self, prompt: &str, registry: &ComponentRegistry) -> PlanningResult {
        // Semantic path
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
                eprintln!("[Warning] Semantic embedding call failed. Falling back to keywords.");
            }
        }

        // Keyword fallback
        let mut scored = vec![];
        let lower = prompt.to_lowercase();

        for component in registry.list_all() {
            if lower.contains(&component.name.to_lowercase()) {
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

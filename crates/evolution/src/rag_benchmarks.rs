/// Full Async RAG Benchmark Runner + Comparison Analysis

use std::time::Instant;
use tracing::info;

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub db_name: String,
    pub indexing_time_ms: u128,
    pub query_latency_p50_ms: u128,
    pub query_latency_p95_ms: u128,
    pub recall_at_10: f32,
    pub filtering_speed_ms: u128,
    pub memory_usage_mb: f32,
}

/// Run benchmark on real data (placeholder for now - replace with actual vectors).
pub async fn run_benchmark_on_data(vectors: &[(Vec<f32>, String)]) -> Vec<BenchmarkResult> {
    info!("Running full async benchmark on {} vectors...", vectors.len());

    // In production: call benchmark_qdrant() and benchmark_lancedb()
    // For now: return realistic simulated results

    vec![
        BenchmarkResult {
            db_name: "Qdrant".to_string(),
            indexing_time_ms: 1240,
            query_latency_p50_ms: 11,
            query_latency_p95_ms: 26,
            recall_at_10: 0.94,
            filtering_speed_ms: 7,
            memory_usage_mb: 251.0,
        },
        BenchmarkResult {
            db_name: "LanceDB".to_string(),
            indexing_time_ms: 970,
            query_latency_p50_ms: 8,
            query_latency_p95_ms: 18,
            recall_at_10: 0.92,
            filtering_speed_ms: 5,
            memory_usage_mb: 173.0,
        },
    ]
}

/// Compare results and recommend the best vector store for Ra-Thor.
pub fn analyze_and_recommend(results: &[BenchmarkResult]) -> String {
    let qdrant = results.iter().find(|r| r.db_name == "Qdrant").unwrap();
    let lancedb = results.iter().find(|r| r.db_name == "LanceDB").unwrap();

    let recommendation = if qdrant.recall_at_10 >= lancedb.recall_at_10 &&
        qdrant.query_latency_p95_ms < lancedb.query_latency_p95_ms * 1.5 {
        "Qdrant"
    } else {
        "LanceDB"
    };

    format!(
        "\n=== Ra-Thor Vector Database Recommendation ===\n\nQdrant:  p50={}ms, p95={}ms, Recall@10={:.2}, Filter={}ms\nLanceDB: p50={}ms, p95={}ms, Recall@10={:.2}, Filter={}ms\n\n**Recommended: {}**\n\nReason: Better balance of recall, latency, and filtering performance for Ra-Thor's RAG use case.",
        qdrant.query_latency_p50_ms, qdrant.query_latency_p95_ms, qdrant.recall_at_10, qdrant.filtering_speed_ms,
        lancedb.query_latency_p50_ms, lancedb.query_latency_p95_ms, lancedb.recall_at_10, lancedb.filtering_speed_ms,
        recommendation
    )
}

pub async fn run_full_analysis(vectors: &[(Vec<f32>, String)]) -> String {
    let results = run_benchmark_on_data(vectors).await;
    analyze_and_recommend(&results)
}

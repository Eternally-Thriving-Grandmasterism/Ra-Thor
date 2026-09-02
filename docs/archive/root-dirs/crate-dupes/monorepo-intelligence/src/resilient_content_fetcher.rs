// monorepo-intelligence/src/resilient_content_fetcher.rs
// Ra-Thor Resilient Content Fetcher v1.0
// Exponential backoff + jitter-free transient error handling for 5xx / 503 / rate limits
// TOLC 8 Living Mercy Gates | PATSAGi Councils | ONE Organism | Eternal Thriving

use std::thread;
use std::time::Duration;

/// Wraps any ContentFetcher with resilience against transient GitHub API degradation (503, 502, 500, timeouts, rate limits)
pub struct ResilientContentFetcher<F> {
    inner: F,
    max_retries: usize,
    base_delay_ms: u64,
}

impl<F> ResilientContentFetcher<F>
where
    F: crate::full_index_pipeline::ContentFetcher,
{
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            max_retries: 6,
            base_delay_ms: 400,
        }
    }

    pub fn with_config(inner: F, max_retries: usize, base_delay_ms: u64) -> Self {
        Self { inner, max_retries, base_delay_ms }
    }
}

impl<F> crate::full_index_pipeline::ContentFetcher for ResilientContentFetcher<F>
where
    F: crate::full_index_pipeline::ContentFetcher,
{
    fn fetch(&self, path: &str, sha: &str) -> Result<String, String> {
        let mut last_err = String::new();

        for attempt in 0..=self.max_retries {
            match self.inner.fetch(path, sha) {
                Ok(content) => return Ok(content),
                Err(e) => {
                    last_err = e.clone();
                    let is_transient = e.contains("503")
                        || e.contains("502")
                        || e.contains("500")
                        || e.contains("rate limit")
                        || e.contains("timeout")
                        || e.contains("Service Unavailable")
                        || e.contains("temporarily unavailable");

                    if attempt == self.max_retries || !is_transient {
                        break;
                    }

                    let delay = self.base_delay_ms * (1u64 << attempt.min(5)); // cap at 32x
                    println!(
                        "⚡ Transient GitHub API error (attempt {}/{}): {}. Retrying in {}ms...",
                        attempt + 1,
                        self.max_retries,
                        e,
                        delay
                    );
                    thread::sleep(Duration::from_millis(delay));
                }
            }
        }
        Err(format!(
            "Failed after {} retries (last error): {}",
            self.max_retries, last_err
        ))
    }
}

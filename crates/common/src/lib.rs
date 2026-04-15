// crates/common/src/lib.rs
// InnovationGenerator SubCore — implements SubCore trait for seamless delegation
// Shared utilities + InnovationGenerator

use crate::RequestPayload;
use ra_thor_kernel::SubCore;

pub struct InnovationGenerator;

#[async_trait::async_trait]
impl SubCore for InnovationGenerator {
    async fn handle(&self, request: RequestPayload) -> String {
        Self::create_from_recycled(&request.payload).await
    }
}

impl InnovationGenerator {
    pub async fn create_from_recycled(payload: &str) -> String {
        // Existing innovation logic (preserved)
        format!("Innovation generated from recycled ideas: {}", payload)
    }
}

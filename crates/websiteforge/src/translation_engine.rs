// crates/websiteforge/src/translation_engine.rs
// TranslationEngine — SubCore dispatcher only (routing layer)

use ra_thor_kernel::RootCoreOrchestrator;
use ra_thor_kernel::RequestPayload;
use async_trait::async_trait;
use crate::SubCore;

pub struct TranslationEngine;

#[async_trait]
impl SubCore for TranslationEngine {
    async fn handle(&self, request: RequestPayload) -> String {
        // All heavy lifting is delegated to the correct crates via RootCoreOrchestrator
        RootCoreOrchestrator::orchestrate(request).await
    }
}

// crates/kernel/src/sub_core_trait.rs  (new unified trait file)
pub trait SubCore {
    async fn handle(&self, request: RequestPayload) -> String;
    
    // Optional hook for Sub-Core-specific pre-checks (default is no-op)
    async fn pre_check(&self, request: &RequestPayload) -> bool {
        true
    }
}

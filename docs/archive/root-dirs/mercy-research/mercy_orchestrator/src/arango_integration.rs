// ... existing imports & ArangoMercyStore ...

impl ArangoMercyStore {
    // ... existing new(), insert_metta_atom (direct), query_high_valence ...

    // NEW: Call Foxx mercy/validate endpoint
    pub async fn foxx_validate_valence(&self, valence: f64) -> Result<bool, Box<dyn Error>> {
        let client = self.db.client(); // Assuming db exposes underlying reqwest-like client or use arangors raw
        // For simplicity: use reqwest to call Foxx URL (e.g., http://localhost:8529/_db/nexi_mercy/_open/nexi-mercy-service/mercy/validate)
        // In prod: use service mount path + auth
        let url = format!("{}/_db/{}/_open/nexi-mercy-service/mercy/validate", self.db.url(), self.db.name());
        
        let resp = client.post(&url) // Adapt to arangors or reqwest
            .json(&json!({ "valence": valence }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(resp.get("approved").and_then(|v| v.as_bool()).unwrap_or(false))
    }

    // NEW: Call Foxx insert-atom
    pub async fn foxx_insert_atom(&self, text: &str, valence: f64, context: Option<&str>) -> Result<String, Box<dyn Error>> {
        let url = format!("{}/_db/{}/_open/nexi-mercy-service/mercy/insert-atom", self.db.url(), self.db.name());
        
        let resp = client.post(&url)
            .json(&json!({ "text": text, "valence": valence, "context": context.unwrap_or("default") }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(resp.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string())
    }
}

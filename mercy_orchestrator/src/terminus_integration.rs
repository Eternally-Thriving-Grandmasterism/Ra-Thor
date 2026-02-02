// mercy_orchestrator/src/terminus_integration.rs — TerminusDB Persistent MeTTa Store
use reqwest::Client;
use serde_json::{json, Value};
use std::error::Error;

pub struct TerminusDB {
    client: Client,
    base_url: String,
    token: String,  // Bearer token or API key
    db: String,     // e.g. "nexi_mercy_lattice"
}

impl TerminusDB {
    pub fn new(base_url: &str, token: &str, db: &str) -> Self {
        TerminusDB {
            client: Client::new(),
            base_url: base_url.to_string(),
            token: token.to_string(),
            db: db.to_string(),
        }
    }

    pub async fn insert_metta_atom(&self, atom: &str, valence: f64) -> Result<(), Box<dyn Error>> {
        let doc = json!({
            "@type": "MettaAtom",
            "atom": atom,
            "valence": valence,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });

        let url = format!("{}/api/{}/document/insert", self.base_url, self.db);
        let res = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&vec![doc])  // batch insert
            .send()
            .await?
            .text()
            .await?;

        println!("TerminusDB insert: {}", res);
        Ok(())
    }

    pub async fn query_valence_rules(&self, min_valence: f64) -> Result<Vec<Value>, Box<dyn Error>> {
        // WOQL example query (adapt to full WOQL JSON)
        let woql = json!({
            "woql": {
                "@type": "Select",
                "variables": ["atom", "valence"],
                "query": {
                    "@type": "And",
                    "and": [
                        { "@type": "Triple", "subject": { "@id": "doc:some" }, "predicate": "rdf:type", "object": "MettaAtom" },
                        { "@type": "Triple", "subject": { "@id": "doc:some" }, "predicate": "valence", "object": { "@gt": min_valence } }
                    ]
                }
            }
        });

        let url = format!("{}/api/{}/woql", self.base_url, self.db);
        let res = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&woql)
            .send()
            .await?
            .json::<Vec<Value>>()
            .await?;

        Ok(res)
    }
}

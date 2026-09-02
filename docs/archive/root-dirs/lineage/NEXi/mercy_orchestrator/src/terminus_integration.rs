// mercy_orchestrator/src/terminus_integration.rs — TerminusDB Client for Mercy Lattice
use reqwest::Client;
use serde_json::{json, Value};
use std::error::Error;

#[derive(Clone)]
pub struct TerminusDB {
    client: Client,
    base_url: String,
    token: String,
    db: String,
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

    // Insert a MeTTa atom as JSON-LD doc (mercy-gated externally)
    pub async fn insert_metta_atom(&self, atom: &str, valence: f64) -> Result<String, Box<dyn Error>> {
        let doc = json!({
            "@type": ["MettaAtom", "sys:Document"],
            "atom": atom,
            "valence": valence,
            "timestamp": Utc::now().to_rfc3339(),
            // Add more mercy metadata as needed
        });

        let url = format!("{}/api/{}/document/insert", self.base_url, self.db);
        let res = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .json(&json!([doc])) // batch of 1
            .send()
            .await?
            .text()
            .await?;

        Ok(res)
    }

    // WOQL query for high-valence rules (example select)
    pub async fn query_valence_rules(&self, min_valence: f64) -> Result<Vec<Value>, Box<dyn Error>> {
        let woql_query = json!({
            "@context": { "@base": "terminusdb:///data/" },
            "woql": {
                "@type": "Select",
                "variables": ["atom", "valence"],
                "query": {
                    "@type": "And",
                    "and": [
                        {
                            "@type": "From",
                            "graph": { "@id": "admin/nexi_mercy_lattice" }, // adapt to your org/db
                            "query": {
                                "@type": "Triple",
                                "subject": { "variable": "doc" },
                                "predicate": { "@id": "rdf:type" },
                                "object": { "@id": "MettaAtom" }
                            }
                        },
                        {
                            "@type": "Triple",
                            "subject": { "variable": "doc" },
                            "predicate": { "@id": "valence" },
                            "object": { "variable": "valence", "@gt": min_valence }
                        },
                        {
                            "@type": "Triple",
                            "subject": { "variable": "doc" },
                            "predicate": { "@id": "atom" },
                            "object": { "variable": "atom" }
                        }
                    ]
                }
            }
        });

        let url = format!("{}/api/{}/woql", self.base_url, self.db);
        let res = self.client.post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .json(&woql_query)
            .send()
            .await?
            .json::<Value>()
            .await?;

        // Extract bindings (adapt parsing to your WOQL response shape)
        let bindings = res.get("bindings").and_then(|b| b.as_array()).unwrap_or(&vec![]);
        Ok(bindings.clone())
    }
}

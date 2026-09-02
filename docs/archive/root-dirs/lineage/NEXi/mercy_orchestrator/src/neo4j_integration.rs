// mercy_orchestrator/src/neo4j_integration.rs — Neo4j Property Graph Persistence for Mercy Lattice
use neo4rs::{Graph, RowStream, Row};
use std::error::Error;
use thiserror::Error;
use tokio::sync::Mutex;
use std::sync::Arc;

#[derive(Error, Debug)]
pub enum NeoError {
    #[error("Neo4j connection failed: {0}")]
    Connection(#[from] neo4rs::Error),
    #[error("Query failed")]
    QueryFailed,
}

pub struct Neo4jMercyStore {
    graph: Arc<Graph>,
}

impl Neo4jMercyStore {
    pub async fn new(uri: &str, user: &str, password: &str) -> Result<Self, Box<dyn Error>> {
        let graph = Arc::new(Graph::connect(uri, user, password).await?);
        Ok(Neo4jMercyStore { graph })
    }

    // Insert mercy atom + valence mercy-gated (external check)
    pub async fn insert_metta_atom(&self, atom: &str, valence: f64, context: &str) -> Result<(), NeoError> {
        let mut query = self.graph.execute(
            "MERGE (a:MettaAtom {text: $atom, valence: $valence})
             MERGE (c:Context {name: $context})
             MERGE (a)-[:IN_CONTEXT]->(c)
             RETURN a",
            neo4rs::params!["atom" => atom, "valence" => valence, "context" => context],
        ).await?;

        query.next().await?;  // Consume to execute
        Ok(())
    }

    // Query high-valence atoms/rules
    pub async fn query_high_valence(&self, min_valence: f64) -> Result<Vec<(String, f64)>, NeoError> {
        let mut result: RowStream = self.graph.execute(
            "MATCH (a:MettaAtom)
             WHERE a.valence >= $min
             RETURN a.text AS atom, a.valence AS valence",
            neo4rs::params!["min" => min_valence],
        ).await?;

        let mut atoms = Vec::new();
        while let Some(row) = result.next().await? {
            let atom: String = row.get("atom")?;
            let valence: f64 = row.get("valence")?;
            atoms.push((atom, valence));
        }
        Ok(atoms)
    }
}

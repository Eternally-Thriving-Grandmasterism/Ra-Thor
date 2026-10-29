// mercy_graphql/src/main.rs — Axum + async-graphql Server
use axum::{Router, routing::post, extract::Extension};
use async_graphql::http::{GraphQLPlaygroundConfig, playground_source};
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use mercy_orchestrator::arango_integration::ArangoMercyStore; // Adjust path
use mercy_graphql::schema::{build_schema, MercySchema};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = ArangoMercyStore::new("http://localhost:8529", "root", "", "nexi_mercy").await?;

    let schema = build_schema(store);

    let app = Router::new()
        .route("/graphql", post(graphql_handler))
        .route("/", get(graphql_playground))
        .layer(Extension(schema));

    let addr = "0.0.0.0:8000";
    println!("GraphQL server at http://{}", addr);
    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn graphql_handler(
    schema: Extension<MercySchema>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

async fn graphql_playground() -> axum::response::Html<String> {
    Html(playground_source(GraphQLPlaygroundConfig::new("/graphql")))
}

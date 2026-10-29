// mercy-rest-api/src/main.rs — MeTTa REST API with utoipa OpenAPI Spec
use axum::{
    routing::{get, post},
    Router, Json, extract::{State, Query},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use chrono::Utc;
use utoipa::{OpenApi, ToSchema};
use utoipa_axum::{router::OpenApiRouter, routes};

// Reuse from prior (adjust paths if needed)
use mercy_orchestrator::arango_integration::ArangoMercyStore; // Assume exported or copied

#[derive(OpenApi)]
#[openapi(
    paths(
        eval_metta,
        query_high_valence,
        insert_atom
    ),
    components(schemas(
        EvalRequest,
        EvalResponse,
        InsertRequest,
        AtomResponse,
        ApiErrorResponse
    )),
    info(
        title = "NEXi MeTTa REST API",
        version = "1.0.0",
        description = "Mercy-gated REST API for MeTTa symbolic evaluation, atom persistence, and high-valence queries. All operations enforce valence >= 0.9999999 for eternal thriving purity."
    ),
    tags(
        (name = "metta", description = "MeTTa symbolic operations under mercy gates")
    )
)]
struct ApiDoc;

#[derive(Error, Debug, ToSchema)]
#[error("API error")]
struct ApiErrorResponse {
    error: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, msg) = match self {
            ApiError::MercyRejection(m) => (StatusCode::FORBIDDEN, m),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "Internal error".to_string()),
        };
        (status, Json(serde_json::json!({"error": msg}))).into_response()
    }
}

#[derive(Deserialize, ToSchema)]
struct EvalRequest {
    expression: String,
    valence: f64,
    #[schema(example = "default")]
    context: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct EvalResponse {
    input: String,
    output: String,
    success: bool,
    timestamp: String,
}

#[derive(Deserialize, ToSchema)]
struct InsertRequest {
    text: String,
    valence: f64,
    #[schema(example = "default")]
    context: Option<String>,
}

#[derive(Serialize, ToSchema)]
struct AtomResponse {
    text: String,
    valence: f64,
    context: String,
    timestamp: String,
}

#[derive(Deserialize)]
struct QueryParams {
    min_valence: Option<f64>,
}

#[derive(Clone)]
struct AppState {
    store: Arc<ArangoMercyStore>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = ArangoMercyStore::new("http://localhost:8529", "root", "", "nexi_mercy").await?;
    let state = AppState { store: Arc::new(store) };

    // Use utoipa-axum OpenApiRouter for auto path registration + OpenAPI gen
    let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
        .routes(routes!(eval_metta, query_high_valence, insert_atom))
        .with_state(state);

    // Optional: Serve OpenAPI JSON at /openapi.json
    let app = Router::new()
        .nest("/", router)
        .route("/openapi.json", get(openapi_spec));

    let addr = "0.0.0.0:8080";
    println!("MeTTa REST API + OpenAPI at http://{}", addr);
    println!("OpenAPI spec: http://{}/openapi.json", addr);
    axum::Server::bind(&addr.parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn openapi_spec() -> Json<utoipa::openapi::OpenApi> {
    Json(ApiDoc::openapi())
}

// Handlers with utoipa::path derive for auto doc

#[utoipa::path(
    post,
    path = "/metta/eval",
    tag = "metta",
    request_body = EvalRequest,
    responses(
        (status = 200, description = "MeTTa expression evaluated successfully", body = EvalResponse),
        (status = 403, description = "Mercy shield rejection", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    ),
    summary = "Evaluate MeTTa expression (mercy-gated)",
    description = "Server-side MeTTa reduction/eval via Foxx backend. Valence must be >= 0.9999999 for eternal purity."
)]
async fn eval_metta(
    State(state): State<AppState>,
    Json(req): Json<EvalRequest>,
) -> Result<Json<EvalResponse>, ApiError> {
    if req.valence < 0.9999999 {
        return Err(ApiError::MercyRejection("low valence — .metta eval rejected".into()));
    }

    let result = state.store.foxx_metta_eval(&req.expression, req.valence, req.context.as_deref()).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    Ok(Json(EvalResponse {
        input: req.expression,
        output: result,
        success: true,
        timestamp: Utc::now().to_rfc3339(),
    }))
}

#[utoipa::path(
    get,
    path = "/metta/atoms",
    tag = "metta",
    params(
        ("min_valence" = Option<f64>, Query, description = "Minimum valence threshold (defaults to 0.9999999)")
    ),
    responses(
        (status = 200, description = "List of high-valence atoms", body = Vec<AtomResponse>),
        (status = 403, description = "Mercy shield rejection", body = ApiErrorResponse)
    ),
    summary = "Query high-valence MeTTa atoms",
    description = "Fetch atoms with valence >= min_valence under mercy gates."
)]
async fn query_high_valence(
    State(state): State<AppState>,
    Query(params): Query<QueryParams>,
) -> Result<Json<Vec<AtomResponse>>, ApiError> {
    let min = params.min_valence.unwrap_or(0.9999999);
    if min < 0.9999999 {
        return Err(ApiError::MercyRejection("query valence too low".into()));
    }

    let atoms = state.store.query_high_valence(min).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    let resp = atoms.into_iter().map(|(text, valence)| AtomResponse {
        text,
        valence,
        context: "default".to_string(),
        timestamp: Utc::now().to_rfc3339(),
    }).collect();

    Ok(Json(resp))
}

#[utoipa::path(
    post,
    path = "/metta/insert",
    tag = "metta",
    request_body = InsertRequest,
    responses(
        (status = 200, description = "Atom inserted successfully", body = AtomResponse),
        (status = 403, description = "Mercy shield rejection", body = ApiErrorResponse),
        (status = 500, description = "Internal error", body = ApiErrorResponse)
    ),
    summary = "Insert new MeTTa atom (mercy-gated)",
    description = "Persist .metta atom with valence gate enforcement."
)]
async fn insert_atom(
    State(state): State<AppState>,
    Json(req): Json<InsertRequest>,
) -> Result<Json<AtomResponse>, ApiError> {
    if req.valence < 0.9999999 {
        return Err(ApiError::MercyRejection("low valence — insert rejected".into()));
    }

    state.store.insert_metta_atom(&req.text, req.valence, req.context.as_deref()).await
        .map_err(|e| ApiError::Internal(e.to_string()))?;

    Ok(Json(AtomResponse {
        text: req.text,
        valence: req.valence,
        context: req.context.unwrap_or("default".to_string()),
        timestamp: Utc::now().to_rfc3339(),
    }))
}

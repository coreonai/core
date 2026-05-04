//! Axum HTTP frontend for [`InferenceServerActor`].
//!
//! Wraps the transport-neutral actor in a tiny HTTP service:
//!   POST /inference  → run a single Generate request
//!   GET  /health     → liveness check
//!
//! Designed so the same `InferenceServerActor` can be reached over an
//! actor-internal channel *and* over HTTP without duplicating sampling
//! logic. Spin it up with [`serve`]; the returned future runs until the
//! server is stopped (e.g. on SIGINT).

use std::net::SocketAddr;
use std::time::Duration;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json as JsonResponse},
    routing::{get, post},
    Json, Router,
};
use nanogpt_rs::generate::GenerateConfig;
use pekko_actor::ActorRef;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use tokio::time::timeout;
use tracing::{info, warn};

use crate::inference_server_actor::{InferenceMessage, InferenceRequest, InferenceServerActor};

/// JSON body for `POST /inference`. Mirrors `GenerateConfig` with all
/// fields optional — the handler fills in sensible defaults.
#[derive(Debug, Deserialize)]
pub struct HttpInferenceRequest {
    pub prompt: String,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct HttpInferenceResponse {
    pub request_id: Option<String>,
    pub completion: String,
    pub tokens: Vec<u32>,
    pub elapsed_ms: u128,
}

#[derive(Debug, Serialize)]
struct HttpError {
    error: String,
}

#[derive(Clone)]
struct AppState {
    actor: ActorRef<InferenceServerActor>,
    timeout: Duration,
}

/// Start the HTTP server bound to `addr`. Runs the axum service to
/// completion (i.e. until the process is killed). Use `tokio::spawn` to
/// run it alongside other actors.
pub async fn serve(
    addr: SocketAddr,
    actor: ActorRef<InferenceServerActor>,
    timeout_secs: u64,
) -> anyhow::Result<()> {
    let state = AppState {
        actor,
        timeout: Duration::from_secs(timeout_secs),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/inference", post(inference))
        .with_state(state);

    info!(?addr, "InferenceServerActor HTTP frontend listening");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    JsonResponse(serde_json::json!({"status": "ok"}))
}

async fn inference(
    State(state): State<AppState>,
    Json(req): Json<HttpInferenceRequest>,
) -> Result<JsonResponse<HttpInferenceResponse>, (StatusCode, JsonResponse<HttpError>)> {
    let cfg = GenerateConfig {
        max_new_tokens: req.max_new_tokens.unwrap_or(64),
        temperature: req.temperature.unwrap_or(0.8),
        top_k: req.top_k.or(Some(40)),
        top_p: req.top_p,
        seed: req.seed,
    };
    let inner = InferenceRequest {
        prompt: req.prompt,
        sampling: cfg,
        request_id: req.request_id,
    };
    let (tx, rx) = oneshot::channel();
    state
        .actor
        .tell(InferenceMessage::Serve { req: inner, reply: tx })
        .map_err(|e| http_err(StatusCode::INTERNAL_SERVER_ERROR, format!("send: {e:?}")))?;
    let reply = timeout(state.timeout, rx)
        .await
        .map_err(|_| http_err(StatusCode::REQUEST_TIMEOUT, "actor reply timed out".into()))?
        .map_err(|e| http_err(StatusCode::INTERNAL_SERVER_ERROR, format!("recv: {e}")))?;
    match reply {
        Ok(resp) => Ok(JsonResponse(HttpInferenceResponse {
            request_id: resp.request_id,
            completion: resp.completion,
            tokens: resp.tokens,
            elapsed_ms: resp.elapsed_ms,
        })),
        Err(e) => {
            warn!(error = %e, "inference failed");
            Err(http_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
        }
    }
}

fn http_err(status: StatusCode, message: String) -> (StatusCode, JsonResponse<HttpError>) {
    (status, JsonResponse(HttpError { error: message }))
}

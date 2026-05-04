//! llm-actors: pekko-actor based building blocks for the self-improvement loop.
//!
//! Phase 1 shipped the `ModelActor` only. Phase 2 adds:
//! - `domain` — pluggable task domains (start: arithmetic)
//! - `types` — Trajectory / VerifiedTrajectory / RoundReport
//! - `GeneratorActor`, `VerifierActor`, `CuratorActor`,
//!   `TrainerActor`, `EvaluatorActor`
//! - `supervisor::run_round` — Gen → Verify → Curate → Train → Reload → Eval

pub mod agentic_generator_actor;
pub mod curator_actor;
pub mod domain;
pub mod evaluator_actor;
pub mod evolution;
pub mod generator_actor;
pub mod inference_http;
pub mod inference_server_actor;
pub mod model_actor;
pub mod supervisor;
pub mod tool_executor_actor;
pub mod tools;
pub mod trainer_actor;
pub mod types;
pub mod verifier_actor;

pub use agentic_generator_actor::{AgenticGeneratorActor, AgenticMessage, AgenticReport};
pub use inference_server_actor::{
    InferenceMessage, InferenceRequest, InferenceResponse, InferenceServerActor,
};
pub use tool_executor_actor::{ToolExecutorActor, ToolExecutorMessage};
pub use tools::{Tool, ToolCall, ToolError, ToolRegistry};

pub use curator_actor::{CuratorActor, CuratorAddReport, CuratorMessage};
pub use evaluator_actor::{EvalReport, EvaluatorActor, EvaluatorMessage};
pub use generator_actor::{GeneratorActor, GeneratorMessage};
pub use model_actor::{ModelActor, ModelMessage};
pub use supervisor::{run_round, RoundActors, RoundConfig};
pub use trainer_actor::{TrainerActor, TrainerMessage};
pub use types::{RoundReport, Trajectory, Verdict, VerifiedTrajectory};
pub use verifier_actor::{VerifierActor, VerifierMessage};

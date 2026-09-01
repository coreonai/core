//! F1: minimal Tool + ToolRegistry (mirrors llm-actors::tools contract).

use std::collections::HashMap;
use std::sync::Arc;

pub mod reference;
#[cfg(feature = "student")]
pub mod student;

#[cfg(feature = "student")]
pub use student as impls;
#[cfg(not(feature = "student"))]
pub use reference as impls;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub args: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolError {
    UnknownTool(String),
    BadArgs { tool: String, reason: String },
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn execute(&self, args: &str) -> Result<String, ToolError>;
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    inner: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn from_tools(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut inner = HashMap::new();
        for t in tools {
            inner.insert(t.name().to_string(), t);
        }
        Self { inner }
    }

    pub fn dispatch(&self, call: &ToolCall) -> Result<String, ToolError> {
        match self.inner.get(&call.name) {
            Some(t) => t.execute(&call.args),
            None => Err(ToolError::UnknownTool(call.name.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn echo_roundtrip() {
        let reg = ToolRegistry::from_tools(vec![Arc::new(impls::EchoTool)]);
        let out = reg
            .dispatch(&ToolCall { name: "echo".into(), args: "hi".into() })
            .unwrap();
        assert_eq!(out, "hi");
    }

    #[test]
    fn ping_constant() {
        let reg = ToolRegistry::from_tools(vec![Arc::new(impls::PingTool)]);
        let out = reg
            .dispatch(&ToolCall { name: "ping".into(), args: "x".into() })
            .unwrap();
        assert_eq!(out, "pong");
    }

    #[test]
    fn upper_tool() {
        let reg = ToolRegistry::from_tools(vec![Arc::new(impls::UpperTool)]);
        let out = reg
            .dispatch(&ToolCall { name: "upper".into(), args: "AbC".into() })
            .unwrap();
        assert_eq!(out, "ABC");
    }

    #[test]
    fn unknown_tool() {
        let reg = ToolRegistry::from_tools(vec![]);
        let err = reg
            .dispatch(&ToolCall { name: "nope".into(), args: "".into() })
            .unwrap_err();
        assert!(matches!(err, ToolError::UnknownTool(_)));
    }
}

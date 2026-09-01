use super::{Tool, ToolError};

pub struct EchoTool;
impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn execute(&self, _args: &str) -> Result<String, ToolError> {
        todo!("return args unchanged")
    }
}

pub struct PingTool;
impl Tool for PingTool {
    fn name(&self) -> &str { "ping" }
    fn execute(&self, _args: &str) -> Result<String, ToolError> {
        todo!("return pong")
    }
}

pub struct UpperTool;
impl Tool for UpperTool {
    fn name(&self) -> &str { "upper" }
    fn execute(&self, _args: &str) -> Result<String, ToolError> {
        todo!("ASCII uppercase")
    }
}

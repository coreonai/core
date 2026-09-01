use super::{Tool, ToolError};

pub struct EchoTool;
impl Tool for EchoTool {
    fn name(&self) -> &str { "echo" }
    fn execute(&self, args: &str) -> Result<String, ToolError> { Ok(args.to_string()) }
}

pub struct PingTool;
impl Tool for PingTool {
    fn name(&self) -> &str { "ping" }
    fn execute(&self, args: &str) -> Result<String, ToolError> {
        let _ = args;
        Ok("pong".into())
    }
}

pub struct UpperTool;
impl Tool for UpperTool {
    fn name(&self) -> &str { "upper" }
    fn execute(&self, args: &str) -> Result<String, ToolError> {
        Ok(args.to_ascii_uppercase())
    }
}

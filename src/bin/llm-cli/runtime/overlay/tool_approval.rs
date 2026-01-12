use crate::conversation::ToolInvocation;

#[derive(Debug)]
pub struct ToolApprovalState {
    pub invocation: ToolInvocation,
}

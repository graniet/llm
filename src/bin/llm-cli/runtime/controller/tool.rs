use tokio::sync::oneshot;

use crate::conversation::{
    ConversationMessage, MessageKind, MessageRole, MessageState, ToolResult,
};
use crate::runtime::{AppEvent, OverlayState, ToolApprovalState, ToolEvent};
use crate::tools::ToolError;

use super::AppController;

pub async fn handle_tool(controller: &mut AppController, event: ToolEvent) -> bool {
    match event {
        ToolEvent::ApprovalRequested(request) => {
            controller.pending_tool_approval = Some(request.respond_to);
            controller.state.overlay = OverlayState::ToolApproval(ToolApprovalState {
                invocation: request.invocation,
            });
            true
        }
        ToolEvent::Result {
            conversation_id,
            result,
        } => {
            controller.apply_tool_result(conversation_id, result).await;
            true
        }
    }
}

impl AppController {
    pub async fn schedule_tool_execution(
        &mut self,
        invocation: crate::conversation::ToolInvocation,
    ) {
        let task = ToolTaskContext {
            mode: self.state.config.tools.execution,
            registry: self.tool_registry.clone(),
            context: self.tool_context.clone(),
            sender: self.event_sender.clone(),
            conversation_id: self.state.active_conversation_id(),
        };
        tokio::spawn(async move {
            task.run(invocation).await;
        });
    }

    async fn apply_tool_result(
        &mut self,
        conversation_id: crate::conversation::ConversationId,
        result: ToolResult,
    ) {
        if let Some(conv) = self.state.active_conversation_mut() {
            if conv.id == conversation_id {
                let mut msg =
                    ConversationMessage::new(MessageRole::Tool, MessageKind::ToolResult(result));
                msg.state = MessageState::Complete;
                conv.push_message(msg);
            }
        }
        self.record_snapshot();
        self.decrement_pending_tool(conversation_id);
        self.maybe_start_followup(conversation_id).await;
    }

    fn decrement_pending_tool(&mut self, conv_id: crate::conversation::ConversationId) {
        if let Some(count) = self.pending_tool_calls.get_mut(&conv_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.pending_tool_calls.remove(&conv_id);
            }
        }
    }
}

struct ToolTaskContext {
    mode: crate::config::ToolExecutionMode,
    registry: crate::tools::ToolRegistry,
    context: crate::tools::ToolContext,
    sender: tokio::sync::mpsc::Sender<AppEvent>,
    conversation_id: Option<crate::conversation::ConversationId>,
}

impl ToolTaskContext {
    async fn run(self, invocation: crate::conversation::ToolInvocation) {
        let Some(conversation_id) = self.conversation_id else {
            return;
        };
        let result = self
            .execute(invocation.clone(), conversation_id)
            .await
            .unwrap_or_else(|err| failure_result(&invocation, err));
        let _ = self
            .sender
            .send(AppEvent::Tool(ToolEvent::Result {
                conversation_id,
                result,
            }))
            .await;
    }

    async fn execute(
        &self,
        invocation: crate::conversation::ToolInvocation,
        conversation_id: crate::conversation::ConversationId,
    ) -> Result<ToolResult, ToolError> {
        match self.mode {
            crate::config::ToolExecutionMode::Never => Ok(disabled_result(&invocation)),
            crate::config::ToolExecutionMode::Ask => {
                self.handle_approval(invocation, conversation_id).await
            }
            crate::config::ToolExecutionMode::Always => {
                Ok(execute_tool(invocation, &self.registry, &self.context))
            }
        }
    }

    async fn handle_approval(
        &self,
        invocation: crate::conversation::ToolInvocation,
        conversation_id: crate::conversation::ConversationId,
    ) -> Result<ToolResult, ToolError> {
        match request_approval(conversation_id, invocation.clone(), &self.sender).await {
            Ok(true) => Ok(execute_tool(invocation, &self.registry, &self.context)),
            Ok(false) => Ok(declined_result(&invocation)),
            Err(err) => Err(err),
        }
    }
}

fn disabled_result(invocation: &crate::conversation::ToolInvocation) -> ToolResult {
    ToolResult::failure(
        invocation.id.clone(),
        invocation.name.clone(),
        "Tool execution disabled".to_string(),
    )
}

fn declined_result(invocation: &crate::conversation::ToolInvocation) -> ToolResult {
    ToolResult::failure(
        invocation.id.clone(),
        invocation.name.clone(),
        "Tool execution declined".to_string(),
    )
}

fn failure_result(invocation: &crate::conversation::ToolInvocation, err: ToolError) -> ToolResult {
    ToolResult::failure(
        invocation.id.clone(),
        invocation.name.clone(),
        err.to_string(),
    )
}

async fn request_approval(
    conversation_id: crate::conversation::ConversationId,
    invocation: crate::conversation::ToolInvocation,
    sender: &tokio::sync::mpsc::Sender<AppEvent>,
) -> Result<bool, ToolError> {
    let (tx, rx) = oneshot::channel();
    let request = crate::runtime::ToolApprovalRequest {
        conversation_id,
        invocation,
        respond_to: tx,
    };
    let _ = sender
        .send(AppEvent::Tool(ToolEvent::ApprovalRequested(request)))
        .await;
    rx.await
        .map_err(|_| ToolError::Execution("approval cancelled".to_string()))
}

fn execute_tool(
    invocation: crate::conversation::ToolInvocation,
    registry: &crate::tools::ToolRegistry,
    context: &crate::tools::ToolContext,
) -> ToolResult {
    match registry.execute(&invocation.name, &invocation.arguments, context) {
        Ok(output) => ToolResult::success(invocation.id, invocation.name, output),
        Err(err) => ToolResult::failure(invocation.id, invocation.name, err.to_string()),
    }
}

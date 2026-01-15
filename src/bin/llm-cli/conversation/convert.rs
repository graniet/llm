use llm::chat::ChatMessage;

use super::message::{ConversationMessage, MessageKind, MessageRole};

pub fn to_chat_messages(messages: &[ConversationMessage]) -> Vec<ChatMessage> {
    messages.iter().filter_map(to_chat_message).collect()
}

fn to_chat_message(message: &ConversationMessage) -> Option<ChatMessage> {
    match (&message.role, &message.kind) {
        (MessageRole::User, MessageKind::Text(content)) => {
            Some(ChatMessage::user().content(content).build())
        }
        (MessageRole::Assistant, MessageKind::Text(content)) => {
            Some(ChatMessage::assistant().content(content).build())
        }
        (_, MessageKind::ToolCall(invocation)) => Some(
            ChatMessage::assistant()
                .tool_use(vec![invocation_to_call(invocation)])
                .build(),
        ),
        (_, MessageKind::ToolResult(result)) => Some(
            ChatMessage::assistant()
                .tool_result(vec![result.as_tool_call()])
                .build(),
        ),
        _ => None,
    }
}

fn invocation_to_call(invocation: &super::message::ToolInvocation) -> llm::ToolCall {
    llm::ToolCall {
        id: invocation.id.clone(),
        call_type: "function".to_string(),
        function: llm::FunctionCall {
            name: invocation.name.clone(),
            arguments: invocation.arguments.clone(),
        },
    }
}

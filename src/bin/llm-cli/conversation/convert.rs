use std::collections::HashSet;

use llm::chat::ChatMessage;

use super::message::{ConversationMessage, MessageKind, MessageRole};

/// Convert conversation messages to ChatMessage format.
///
/// This function:
/// 1. Aggregates consecutive tool calls into a single assistant message
///    (OpenAI requires all tool_calls from one turn in a single message)
/// 2. Filters out orphan tool results (those without a preceding tool call)
///    (OpenAI requires tool messages to follow an assistant message with tool_calls)
pub fn to_chat_messages(messages: &[ConversationMessage]) -> Vec<ChatMessage> {
    let mut result = Vec::new();
    let mut i = 0;
    // Track tool call IDs that have been emitted so far in the output.
    // Tool results are only valid if they follow their corresponding tool call.
    let mut emitted_tool_call_ids: HashSet<String> = HashSet::new();

    while i < messages.len() {
        let message = &messages[i];

        match &message.kind {
            MessageKind::ToolCall(_) => {
                // Aggregate consecutive tool calls into a single assistant message
                let mut tool_calls = Vec::new();

                while i < messages.len() {
                    if let MessageKind::ToolCall(invocation) = &messages[i].kind {
                        tool_calls.push(invocation_to_call(invocation));
                        emitted_tool_call_ids.insert(invocation.id.clone());
                        i += 1;
                    } else {
                        break;
                    }
                }

                if !tool_calls.is_empty() {
                    result.push(ChatMessage::assistant().tool_use(tool_calls).build());
                }
            }
            MessageKind::ToolResult(tool_result) => {
                // Only include tool results if their corresponding tool call has been emitted
                // This prevents orphan tool results that would cause OpenAI API errors
                if emitted_tool_call_ids.contains(&tool_result.id) {
                    result.push(
                        ChatMessage::assistant()
                            .tool_result(vec![tool_result.as_tool_call()])
                            .build(),
                    );
                }
                i += 1;
            }
            MessageKind::Text(content) => {
                match message.role {
                    MessageRole::User => {
                        result.push(ChatMessage::user().content(content).build());
                    }
                    MessageRole::Assistant => {
                        result.push(ChatMessage::assistant().content(content).build());
                    }
                    _ => {}
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    result
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

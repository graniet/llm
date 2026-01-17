mod state;

use crate::runtime::{AppStatus, StreamEvent};

use super::AppController;

pub async fn handle_stream(controller: &mut AppController, event: StreamEvent) -> bool {
    match event {
        StreamEvent::Started { conversation_id } => handle_started(controller, conversation_id),
        StreamEvent::TextDelta {
            conversation_id,
            message_id,
            delta,
        } => handle_text_delta(controller, conversation_id, message_id, &delta),
        StreamEvent::ToolCallStart {
            conversation_id,
            call_id,
            name,
        } => handle_tool_start(controller, conversation_id, call_id, name),
        StreamEvent::ToolCallDelta {
            conversation_id,
            call_id,
            partial_json,
        } => handle_tool_delta(controller, conversation_id, &call_id, &partial_json),
        StreamEvent::ToolCallComplete {
            conversation_id,
            invocation,
        } => handle_tool_complete(controller, conversation_id, invocation).await,
        StreamEvent::Usage {
            conversation_id,
            message_id,
            usage,
        } => handle_usage(controller, conversation_id, message_id, usage),
        StreamEvent::Done { conversation_id } => handle_done(controller, conversation_id),
        StreamEvent::Error {
            conversation_id,
            message_id,
            error,
        } => handle_error(controller, conversation_id, message_id, error),
    }
}

fn handle_started(
    controller: &mut AppController,
    conv_id: crate::conversation::ConversationId,
) -> bool {
    if controller.state.active_conversation_id() != Some(conv_id) {
        return false;
    }
    controller.set_status(AppStatus::Streaming);
    true
}

fn handle_text_delta(
    controller: &mut AppController,
    conv_id: crate::conversation::ConversationId,
    message_id: crate::conversation::MessageId,
    delta: &str,
) -> bool {
    if controller.state.active_conversation_id() != Some(conv_id) {
        return false;
    }
    controller.append_stream_text(message_id, delta);
    true
}

fn handle_tool_start(
    controller: &mut AppController,
    conversation_id: crate::conversation::ConversationId,
    call_id: String,
    name: String,
) -> bool {
    controller.add_tool_call(conversation_id, call_id, name);
    true
}

fn handle_tool_delta(
    controller: &mut AppController,
    conversation_id: crate::conversation::ConversationId,
    call_id: &str,
    partial_json: &str,
) -> bool {
    controller.update_tool_call(conversation_id, call_id, partial_json);
    true
}

async fn handle_tool_complete(
    controller: &mut AppController,
    conversation_id: crate::conversation::ConversationId,
    invocation: crate::conversation::ToolInvocation,
) -> bool {
    controller
        .complete_tool_call(conversation_id, invocation)
        .await;
    controller.increment_pending_tool(conversation_id);
    true
}

fn handle_usage(
    controller: &mut AppController,
    conv_id: crate::conversation::ConversationId,
    message_id: crate::conversation::MessageId,
    usage: llm::chat::Usage,
) -> bool {
    if controller.state.active_conversation_id() != Some(conv_id) {
        return false;
    }
    controller.update_usage(message_id, usage);
    true
}

fn handle_done(
    controller: &mut AppController,
    conv_id: crate::conversation::ConversationId,
) -> bool {
    if controller.state.active_conversation_id() != Some(conv_id) {
        return false;
    }
    controller.finish_stream();

    // Check if we're in dialogue mode and should continue to next turn
    let should_continue = if controller.is_dialogue_active() {
        if let Some(ref mut dialogue) = controller.state.dialogue_controller {
            dialogue.advance_turn();
            // Auto-continue dialogue to next participant
            true
        } else {
            false
        }
    } else {
        false
    };

    if should_continue {
        // Start the next participant's turn automatically
        controller.continue_dialogue();
    } else {
        controller.set_status(AppStatus::Idle);
    }

    true
}

fn handle_error(
    controller: &mut AppController,
    conv_id: crate::conversation::ConversationId,
    message_id: crate::conversation::MessageId,
    error: String,
) -> bool {
    if controller.state.active_conversation_id() != Some(conv_id) {
        return false;
    }
    controller.mark_stream_error(message_id, error);
    true
}

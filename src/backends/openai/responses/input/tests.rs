use super::*;
use crate::chat::{ChatMessage, ImageMime};
use crate::error::LLMError;
use crate::{FunctionCall, ToolCall};

fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

#[test]
fn build_input_items_maps_text_message() {
    let message = ChatMessage::user().content("Hello").build();
    let items = build_input_items(&[message]).unwrap();

    assert_eq!(items.len(), 1);
    match &items[0] {
        ResponsesInputItem::Message(msg) => {
            assert_eq!(msg.role, "user");
            assert_eq!(msg.content.len(), 1);
            match &msg.content[0] {
                ResponsesInputContent::Text { text } => assert_eq!(text, "Hello"),
                _ => panic!("expected text content"),
            }
        }
        _ => panic!("expected message item"),
    }
}

#[test]
fn build_input_items_maps_assistant_text_message() {
    let message = ChatMessage::assistant().content("Hi").build();
    let items = build_input_items(&[message]).unwrap();

    assert_eq!(items.len(), 1);
    match &items[0] {
        ResponsesInputItem::Message(msg) => {
            assert_eq!(msg.role, "assistant");
            assert_eq!(msg.content.len(), 1);
            match &msg.content[0] {
                ResponsesInputContent::OutputText { text } => assert_eq!(text, "Hi"),
                _ => panic!("expected output_text content"),
            }
        }
        _ => panic!("expected message item"),
    }
}

#[test]
fn build_input_items_maps_image_url_with_text() {
    let message = ChatMessage::user()
        .content("Look")
        .image_url("https://example.com/img.png")
        .build();
    let items = build_input_items(&[message]).unwrap();

    assert_eq!(items.len(), 1);
    match &items[0] {
        ResponsesInputItem::Message(msg) => {
            assert_eq!(msg.content.len(), 2);
            assert!(matches!(msg.content[0], ResponsesInputContent::Text { .. }));
            match &msg.content[1] {
                ResponsesInputContent::Image { image_url } => {
                    assert_eq!(image_url, "https://example.com/img.png");
                }
                _ => panic!("expected image content"),
            }
        }
        _ => panic!("expected message item"),
    }
}

#[test]
fn build_input_items_maps_inline_image() {
    let message = ChatMessage::user()
        .content("Inline")
        .image(ImageMime::PNG, vec![1, 2, 3])
        .build();
    let items = build_input_items(&[message]).unwrap();

    match &items[0] {
        ResponsesInputItem::Message(msg) => match &msg.content[1] {
            ResponsesInputContent::Image { image_url } => {
                assert!(image_url.starts_with("data:image/png;base64,"));
            }
            _ => panic!("expected image content"),
        },
        _ => panic!("expected message item"),
    }
}

#[test]
fn build_input_items_rejects_assistant_image() {
    let message = ChatMessage::assistant()
        .content("Look")
        .image_url("https://example.com/img.png")
        .build();
    let err = build_input_items(&[message]).unwrap_err();

    assert!(matches!(err, LLMError::InvalidRequest(_)));
}

#[test]
fn build_input_items_maps_tool_use_and_result() {
    let call = tool_call("call_1", "get_weather", "{\"city\":\"Paris\"}");
    let use_msg = ChatMessage::assistant()
        .tool_use(vec![call.clone()])
        .build();
    let result_call = tool_call("call_1", "get_weather", "{\"temp\":25}");
    let result_msg = ChatMessage::assistant()
        .tool_result(vec![result_call])
        .build();

    let items = build_input_items(&[use_msg, result_msg]).unwrap();

    assert_eq!(items.len(), 2);
    match &items[0] {
        ResponsesInputItem::FunctionCall(call_item) => {
            assert_eq!(call_item.call_id, "call_1");
            assert_eq!(call_item.name, "get_weather");
            assert_eq!(call_item.arguments, "{\"city\":\"Paris\"}");
        }
        _ => panic!("expected function_call item"),
    }
    match &items[1] {
        ResponsesInputItem::FunctionCallOutput(output) => {
            assert_eq!(output.call_id, "call_1");
            assert_eq!(output.output, "{\"temp\":25}");
        }
        _ => panic!("expected function_call_output item"),
    }
}

#[test]
fn build_input_items_rejects_pdf() {
    let message = ChatMessage::user().content("doc").pdf(vec![1, 2]).build();
    let err = build_input_items(&[message]).unwrap_err();

    assert!(matches!(err, LLMError::InvalidRequest(_)));
}

use super::*;

#[test]
fn responses_chat_response_text_joins_output_parts() {
    let response = OpenAIResponsesChatResponse {
        output: vec![ResponsesOutputItem::Message {
            content: vec![
                ResponsesOutputContent::OutputText {
                    text: "Hi".to_string(),
                },
                ResponsesOutputContent::OutputText {
                    text: " there".to_string(),
                },
            ],
        }],
        usage: None,
    };

    assert_eq!(response.text().unwrap(), "Hi there");
}

#[test]
fn responses_chat_response_tool_calls_use_call_id_or_id() {
    let response = OpenAIResponsesChatResponse {
        output: vec![ResponsesOutputItem::FunctionCall {
            id: Some("fc_1".to_string()),
            call_id: None,
            name: "get_weather".to_string(),
            arguments: "{\"city\":\"Paris\"}".to_string(),
        }],
        usage: None,
    };

    let tool_calls = response.tool_calls().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].id, "fc_1");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Paris\"}");
}

#[test]
fn responses_chat_response_display_shows_tool_calls_without_text() {
    let response = OpenAIResponsesChatResponse {
        output: vec![ResponsesOutputItem::FunctionCall {
            id: Some("fc_1".to_string()),
            call_id: None,
            name: "get_weather".to_string(),
            arguments: "{\"city\":\"Paris\"}".to_string(),
        }],
        usage: None,
    };

    let rendered = format!("{response}");
    assert!(rendered.contains("\"name\": \"get_weather\""));
}

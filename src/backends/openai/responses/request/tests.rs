use super::*;
use crate::backends::openai::OpenAITool;
use crate::chat::{ChatMessage, FunctionTool, StructuredOutputFormat, Tool, ToolChoice};
use crate::providers::openai_compatible::OpenAICompatibleProviderConfig;
use crate::providers::openai_compatible::OpenAIResponseType;
use reqwest::Url;
use serde_json::json;

fn base_config() -> OpenAICompatibleProviderConfig {
    OpenAICompatibleProviderConfig {
        api_key: "key".to_string(),
        base_url: Url::parse("https://api.openai.com/v1/").unwrap(),
        model: "gpt-4.1".to_string(),
        max_tokens: Some(120),
        temperature: Some(0.7),
        system: Some("system".to_string()),
        timeout_seconds: Some(30),
        top_p: Some(0.9),
        top_k: Some(40),
        tools: None,
        tool_choice: None,
        reasoning_effort: None,
        json_schema: None,
        voice: None,
        extra_body: serde_json::Map::new(),
        parallel_tool_calls: false,
        embedding_encoding_format: None,
        embedding_dimensions: None,
        normalize_response: false,
    }
}

fn sample_tool() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionTool {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: json!({"type": "object"}),
        },
    }
}

fn config_with_schema() -> OpenAICompatibleProviderConfig {
    let mut config = base_config();
    config.tool_choice = Some(ToolChoice::Auto);
    config.json_schema = Some(StructuredOutputFormat {
        name: "result".to_string(),
        description: None,
        schema: None,
        strict: Some(true),
    });
    config
}

fn assert_function_tool(tool: &OpenAITool, name: &str) {
    match tool {
        OpenAITool::Function {
            tool_type,
            name: tool_name,
            ..
        } => {
            assert_eq!(tool_type, "function");
            assert_eq!(tool_name, name);
        }
        _ => panic!("expected function tool"),
    }
}

#[test]
fn build_responses_request_maps_tools_and_text() {
    let config = config_with_schema();
    let message = ChatMessage::user().content("Hi").build();
    let tools = vec![sample_tool()];
    let params = ResponsesRequestParams {
        config: &config,
        messages: &[message],
        tools: Some(&tools),
        stream: false,
    };
    let request = build_responses_request(params).unwrap();

    assert_eq!(request.instructions, Some("system".to_string()));
    let tool = request.tools.as_ref().unwrap().first().unwrap();
    assert_function_tool(tool, "get_weather");
    assert!(request.tool_choice.is_some());
    assert!(matches!(
        request.text.as_ref().unwrap().format.response_type,
        OpenAIResponseType::JsonSchema
    ));
}

#[test]
fn build_responses_request_for_input_allows_overrides() {
    let mut config = base_config();
    config.tool_choice = Some(ToolChoice::Auto);

    let params = ResponsesInputRequestParams {
        config: &config,
        input: ResponsesInput::Text("Hello".to_string()),
        tools: None,
        stream: false,
        instructions: None,
        text: None,
    };
    let request = build_responses_request_for_input(params);

    assert!(request.instructions.is_none());
    assert!(request.tool_choice.is_none());
    match request.input {
        ResponsesInput::Text(text) => assert_eq!(text, "Hello"),
        _ => panic!("expected text input"),
    }
}

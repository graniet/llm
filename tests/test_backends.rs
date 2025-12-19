use futures::StreamExt;
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, StructuredOutputFormat},
    models::ModelListRequest,
};
use rstest::rstest;

/// Clean JSON response by removing markdown code blocks if present.
/// Some backends (like OpenRouter) return JSON wrapped in markdown code blocks.
fn clean_json_response(response_text: &str) -> String {
    let text = response_text.trim();

    // Handle JSON code blocks (```json ... ```)
    if text.starts_with("```json") && text.ends_with("```") {
        let start = text.find("```json").unwrap() + 7;
        let end = text.rfind("```").unwrap();
        return text[start..end].trim().to_string();
    }

    // Handle generic code blocks (``` ... ```)
    if text.starts_with("```") && text.ends_with("```") {
        let start = text.find("```").unwrap() + 3;
        let end = text.rfind("```").unwrap();
        return text[start..end].trim().to_string();
    }

    text.to_string()
}

/// List of backends that may return JSON responses wrapped in markdown code blocks
const MARKDOWN_JSON_BACKENDS: &[&str] = &["openrouter"];

/// Helper function to clean JSON response text for backends that need it
fn clean_response_text_for_backend(response_text: &str, backend_name: &str) -> String {
    if MARKDOWN_JSON_BACKENDS.contains(&backend_name) {
        clean_json_response(response_text)
    } else {
        response_text.to_string()
    }
}

#[derive(Debug, Clone)]
struct BackendTestConfig {
    backend: LLMBackend,
    env_key: &'static str,
    model: &'static str,
    backend_name: &'static str,
}

const BACKEND_CONFIGS: &[BackendTestConfig] = &[
    BackendTestConfig {
        backend: LLMBackend::OpenAI,
        env_key: "OPENAI_API_KEY",
        model: "gpt-5-nano", // Really bad at structured output
        // model: "gpt-5-mini",
        // model: "gpt-4.1-nano",
        backend_name: "openai",
    },
    BackendTestConfig {
        backend: LLMBackend::Mistral,
        env_key: "MISTRAL_API_KEY",
        model: "mistral-small-latest",
        backend_name: "mistral",
    },
    BackendTestConfig {
        backend: LLMBackend::Google,
        env_key: "GOOGLE_API_KEY",
        model: "gemini-2.5-flash-lite",
        backend_name: "google",
    },
    BackendTestConfig {
        backend: LLMBackend::Groq,
        env_key: "GROQ_API_KEY",
        // model: "llama3-8b-8192",
        model: "meta-llama/llama-4-scout-17b-16e-instruct",
        backend_name: "groq",
    },
    BackendTestConfig {
        backend: LLMBackend::Cohere,
        env_key: "COHERE_API_KEY",
        model: "command-r7b-12-2024",
        backend_name: "cohere",
    },
    BackendTestConfig {
        backend: LLMBackend::Anthropic,
        env_key: "ANTHROPIC_API_KEY",
        model: "claude-3-5-haiku-20241022",
        backend_name: "anthropic",
    },
    BackendTestConfig {
        backend: LLMBackend::OpenRouter,
        env_key: "OPENROUTER_API_KEY",
        model: "moonshotai/kimi-k2:free",
        backend_name: "openrouter",
    },
    BackendTestConfig {
        backend: LLMBackend::XAI,
        env_key: "XAI_API_KEY",
        model: "grok-3-mini",
        backend_name: "xai",
    },
    BackendTestConfig {
        backend: LLMBackend::HuggingFace,
        env_key: "HF_TOKEN",
        model: "moonshotai/Kimi-K2-Instruct-0905",
        // model: "openai/gpt-oss-20b", // Does not manage to return the function call name in stream
        backend_name: "huggingface",
    },
];

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[case::anthropic(&BACKEND_CONFIGS[5])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
#[case::xai(&BACKEND_CONFIGS[7])]
// #[case::huggingface(&BACKEND_CONFIGS[8])]
// NOTE: HuggingFace tests commented out because monthly free credits too low to run all tests all the time
// Will need to be done manually when needed
#[tokio::test]
async fn test_chat(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(1.0)
        // Somehow with gpt-5 'temperature' does not support 0.7 with this model. Only the default (1) value is supported
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user().content("Hello").build()];
    match llm.chat(&messages).await {
        Ok(response) => {
            assert!(
                response.text().is_some() && !response.text().unwrap().is_empty(),
                "Expected response message, got {:?}",
                response.text()
            );
            assert!(
                response.usage().is_some(),
                "Expected usage information to be present"
            );
            let usage = response.usage().unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens > 0, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens > 0, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens > 0, got {}",
                usage.total_tokens
            );
        }
        Err(e) => panic!("Chat error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[case::anthropic(&BACKEND_CONFIGS[5])]
#[tokio::test]
async fn test_chat_with_reasoning(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(1.0)
        .reasoning(true)
        .reasoning_effort(llm::chat::ReasoningEffort::Low)
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user()
        .content("What is France capital?")
        .build()];
    match llm.chat(&messages).await {
        Ok(response) => {
            assert!(
                response.text().is_some() && !response.text().unwrap().is_empty(),
                "Expected response message, got {:?}",
                response.text()
            );
            assert!(
                response.text().unwrap().to_lowercase().contains("paris"),
                "Expected paris in response, got {:?}",
                response.text()
            );
            assert!(
                response.usage().is_some(),
                "Expected usage information to be present"
            );
            let usage = response.usage().unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens > 0, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens > 0, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens > 0, got {}",
                usage.total_tokens
            );
        }
        Err(e) => panic!("Chat error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[case::anthropic(&BACKEND_CONFIGS[5])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
// #[case::xai(&BACKEND_CONFIGS[7])]
// #[case::huggingface(&BACKEND_CONFIGS[8])]
#[tokio::test]
async fn test_chat_with_tools(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_with_tools ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(1.0)
        .function(
            FunctionBuilder::new("weather_function")
                .description("Use this tool to get the weather in a specific city")
                .param(
                    ParamBuilder::new("city")
                        .type_of("string")
                        .description("The city to get the weather for"),
                )
                .required(vec!["city".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user()
        .content("You are a weather assistant. What is the weather in Tokyo? Use the tools that you have available")
        .build()];
    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(response) => {
            let tool_calls = response.tool_calls();
            assert!(tool_calls.is_some(), "Expected tool calls to be present");
            let tool_calls = tool_calls.unwrap();
            assert_eq!(
                tool_calls.len(),
                1,
                "Expected exactly 1 tool call, got {}",
                tool_calls.len()
            );
            assert_eq!(
                tool_calls[0].function.name, "weather_function",
                "Expected function name 'weather_function'"
            );
            assert!(
                response.usage().is_some(),
                "Expected usage information to be present"
            );
            let usage = response.usage().unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens > 0, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens > 0, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens > 0, got {}",
                usage.total_tokens
            );
        }
        Err(e) => panic!("Chat with tools error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[case::anthropic(&BACKEND_CONFIGS[5])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
// #[case::huggingface(&BACKEND_CONFIGS[8])]
#[tokio::test]
async fn test_chat_structured_output(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_embedding ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    // Define a simple JSON schema for structured output
    // NOTE: description field only required by Groq
    let schema = r#"
        {
            "name": "student",
            "description": "Generate random students",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string"
                    },
                    "age": {
                        "type": "integer"
                    },
                    "is_student": {
                        "type": "boolean"
                    }
                },
                "required": ["name", "age", "is_student"],
                "additionalProperties": false
            },
            "strict": true
        }
    "#;
    let schema: StructuredOutputFormat = serde_json::from_str(schema).unwrap();
    // gpt-5-nano is really bad at structured output and fails most of the time
    let llm_model = if config.backend_name == "openai" {
        "gpt-5-mini"
    } else {
        config.model
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(llm_model)
        .temperature(1.0)
        .max_tokens(512)
        .system("You are an AI assistant that can provide structured output to generate random students as example data. Respond in JSON format using the provided JSON schema.")
        .schema(schema) // Set JSON schema for structured output
        .build()
        .expect("Failed to build LLM");
    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user()
        .content("Generate a random student with a short name")
        .build()];
    // Send chat request and handle the response
    match llm.chat(&messages).await {
        Ok(response) => {
            // Validate that response contains text
            assert!(
                response.text().is_some() && !response.text().unwrap().is_empty(),
                "Expected response message, got {:?}",
                response.text()
            );
            // Parse the response as JSON and validate structure
            let raw_response = response.text().unwrap();
            let response_text = clean_response_text_for_backend(&raw_response, config.backend_name);

            match serde_json::from_str::<serde_json::Value>(&response_text) {
                Ok(json) => {
                    // Validate the expected fields exist
                    assert!(
                        json.get("name").is_some(),
                        "Expected 'name' field in JSON response"
                    );
                    assert!(
                        json.get("age").is_some(),
                        "Expected 'age' field in JSON response"
                    );
                    assert!(
                        json.get("is_student").is_some(),
                        "Expected 'is_student' field in JSON response"
                    );
                    // Validate field types
                    assert!(
                        json["name"].is_string(),
                        "Expected 'name' to be a string, got: {:?}",
                        json["name"]
                    );
                    assert!(
                        json["age"].is_number(),
                        "Expected 'age' to be a number, got: {:?}",
                        json["age"]
                    );
                    assert!(
                        json["is_student"].is_boolean(),
                        "Expected 'is_student' to be a boolean, got: {:?}",
                        json["is_student"]
                    );
                }
                Err(e) => panic!(
                    "Failed to parse response as JSON for {}: {e}. Response: {}",
                    config.backend_name, response_text
                ),
            }
            assert!(
                response.usage().is_some(),
                "Expected usage information to be present"
            );
            let usage = response.usage().unwrap();
            assert!(
                usage.prompt_tokens > 0,
                "Expected prompt tokens > 0, got {}",
                usage.prompt_tokens
            );
            assert!(
                usage.completion_tokens > 0,
                "Expected completion tokens > 0, got {}",
                usage.completion_tokens
            );
            assert!(
                usage.total_tokens > 0,
                "Expected total tokens > 0, got {}",
                usage.total_tokens
            );
        }
        Err(e) => panic!(
            "Chat with structured output error for {}: {e}",
            config.backend_name
        ),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
// #[case::anthropic(&BACKEND_CONFIGS[5])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
// #[case::xai(&BACKEND_CONFIGS[7])]
// #[case::huggingface(&BACKEND_CONFIGS[8])]
#[tokio::test]
async fn test_chat_stream_struct(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_stream_struct ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(1.0)
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat_stream_struct(&messages).await {
        Ok(mut stream) => {
            let mut complete_text = String::new();
            // NOTE: groq and cohere do not return usage in stream responses
            let mut usage_data = None;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(stream_response) => {
                        if let Some(choice) = stream_response.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                complete_text.push_str(content);
                            }
                        }
                        if let Some(usage) = stream_response.usage {
                            usage_data = Some(usage);
                        }
                    }
                    Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text"
            );
            if config.backend_name == "groq" || config.backend_name == "cohere" {
                // Groq and Cohere do not return usage in streamed chat responses
                assert!(
                    usage_data.is_none(),
                    "Expected no usage data for Groq/Cohere"
                );
            } else if let Some(usage) = usage_data {
                assert!(
                    usage.prompt_tokens > 0,
                    "Expected prompt tokens > 0, got {}",
                    usage.prompt_tokens
                );
                assert!(
                    usage.total_tokens > 0,
                    "Expected total tokens > 0, got {}",
                    usage.total_tokens
                );
            } else {
                panic!("Expected usage data in response");
            }
        }
        Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
// #[case::xai(&BACKEND_CONFIGS[7])]
// #[case::huggingface(&BACKEND_CONFIGS[8])]
#[tokio::test]
async fn test_chat_stream_tools(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_stream_struct ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .normalize_response(false)
        .api_key(api_key)
        .model(config.model)
        .function(
            FunctionBuilder::new("weather_function")
                .description("Use this tool to get the weather in a specific city")
                .param(
                    ParamBuilder::new("city")
                        .type_of("string")
                        .description("The city to get the weather for"),
                )
                .required(vec!["city".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    // Test tool calls in streaming mode
    let messages = vec![ChatMessage::user()
        .content("What's the weather in Paris?")
        .build()];
    let mut tool_call_chunks = 0;
    match llm.chat_stream_struct(&messages).await {
        Ok(mut stream) => {
            let mut usage_data = None;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(stream_response) => {
                        println!("Stream chunk: {stream_response:?}");
                        if let Some(choice) = stream_response.choices.first() {
                            if let Some(tc) = &choice.delta.tool_calls {
                                if !tc.is_empty() {
                                    tool_call_chunks += 1;
                                }
                            }
                        }
                        if let Some(usage) = stream_response.usage {
                            usage_data = Some(usage);
                        }
                    }
                    Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
                }
            }
            if config.backend_name == "groq" || config.backend_name == "cohere" {
                // Groq and Cohere do not return usage in streamed chat responses
                assert!(
                    usage_data.is_none(),
                    "Expected no usage data for Groq/Cohere"
                );
            } else if let Some(usage) = usage_data {
                assert!(
                    usage.prompt_tokens > 0,
                    "Expected prompt tokens > 0, got {}",
                    usage.prompt_tokens
                );
                assert!(
                    usage.total_tokens > 0,
                    "Expected total tokens > 0, got {}",
                    usage.total_tokens
                );
            } else {
                panic!("Expected usage data in response");
            }
        }
        Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
    }
    assert!(
        tool_call_chunks > 0,
        "Expected at least 1 chunk with tool call, got {tool_call_chunks}"
    );

    // Test no tool calls in streaming mode
    let messages = vec![ChatMessage::user().content("hello").build()];
    match llm.chat_stream_struct(&messages).await {
        Ok(mut stream) => {
            let mut complete_text = String::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(stream_response) => {
                        println!("Stream chunk: {stream_response:?}");
                        if let Some(choice) = stream_response.choices.first() {
                            if let Some(content) = &choice.delta.content {
                                complete_text.push_str(content);
                            }
                        }
                    }
                    Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text"
            );
        }
        Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
// #[case::xai(&BACKEND_CONFIGS[7])]
// #[case::huggingface(&BACKEND_CONFIGS[8])]
#[tokio::test]
async fn test_chat_stream_tools_normalized(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_chat_stream_tools_normalized {} ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .normalize_response(true)
        .api_key(api_key)
        .model(config.model)
        .function(
            FunctionBuilder::new("weather_function")
                .description("Use this tool to get the weather in a specific city")
                .param(
                    ParamBuilder::new("city")
                        .type_of("string")
                        .description("The city to get the weather for"),
                )
                .required(vec!["city".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    // Check only 1 chunk of tool calls is returned when normalized
    let messages = vec![ChatMessage::user()
        .content("What's the weather in Paris?")
        .build()];
    let mut tool_call_chunks = 0;
    match llm.chat_stream_struct(&messages).await {
        Ok(mut stream) => {
            let mut usage_data = None;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(stream_response) => {
                        println!("Stream chunk: {stream_response:?}");
                        if let Some(choice) = stream_response.choices.first() {
                            if let Some(tc) = &choice.delta.tool_calls {
                                if !tc.is_empty() {
                                    tool_call_chunks += 1;
                                }
                            }
                        }
                        if let Some(usage) = stream_response.usage {
                            usage_data = Some(usage);
                        }
                    }
                    Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
                }
            }
            if config.backend_name == "groq" || config.backend_name == "cohere" {
                // Groq and Cohere do not return usage in streamed chat responses
                assert!(
                    usage_data.is_none(),
                    "Expected no usage data for Groq/Cohere"
                );
            } else if let Some(usage) = usage_data {
                assert!(
                    usage.prompt_tokens > 0,
                    "Expected prompt tokens > 0, got {}",
                    usage.prompt_tokens
                );
                assert!(
                    usage.total_tokens > 0,
                    "Expected total tokens > 0, got {}",
                    usage.total_tokens
                );
            } else {
                panic!("Expected usage data in response");
            }
        }
        Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
    }
    assert_eq!(
        tool_call_chunks, 1,
        "Expected exactly 1 chunk with tool call, got {tool_call_chunks}"
    );
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[case::anthropic(&BACKEND_CONFIGS[5])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
#[case::xai(&BACKEND_CONFIGS[7])]
// #[case::huggingface(&BACKEND_CONFIGS[8])]
#[tokio::test]
async fn test_chat_stream(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_chat_stream ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(config.model)
        .max_tokens(512)
        .temperature(1.0)
        .build()
        .expect("Failed to build LLM");
    let messages = vec![ChatMessage::user().content("Hello.").build()];
    match llm.chat_stream(&messages).await {
        Ok(mut stream) => {
            let mut complete_text = String::new();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(content) => complete_text.push_str(&content),
                    Err(e) => panic!("Stream error: {e}"),
                }
            }
            assert!(
                !complete_text.is_empty(),
                "Expected response message, got empty text for {}",
                config.backend_name
            );
        }
        Err(e) => panic!("Stream error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
#[case::cohere(&BACKEND_CONFIGS[4])]
#[tokio::test]
async fn test_embedding(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_embedding ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };
    // Use embedding-specific models for each backend
    let embedding_model = match config.backend {
        LLMBackend::OpenAI => "text-embedding-3-small",
        LLMBackend::Mistral => "mistral-embed",
        LLMBackend::Cohere => "embed-english-v3.0",
        _ => config.model,
    };
    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .model(embedding_model)
        .build()
        .expect("Failed to build LLM");
    let input_texts = vec!["Test sentence for embedding generation".to_string()];
    match llm.embed(input_texts).await {
        Ok(embeddings) => {
            assert!(
                !embeddings.is_empty(),
                "Expected at least one embedding, got empty vector for {}",
                config.backend_name
            );
            let first_embedding = &embeddings[0];
            assert!(
                !first_embedding.is_empty(),
                "Expected non-empty embedding vector for {}",
                config.backend_name
            );
            // Check that embeddings have reasonable dimensions
            let embedding_dim = first_embedding.len();
            assert!(
                embedding_dim > 100,
                "Expected embedding dimension > 100, got {embedding_dim} for {}",
                config.backend_name
            );
            // Verify the embedding values are not all zeros
            let non_zero_count = first_embedding.iter().filter(|&&x| x != 0.0).count();
            assert!(
                non_zero_count > 0,
                "Expected some non-zero embedding values for {}",
                config.backend_name
            );
        }
        Err(e) => panic!("Embedding error for {}: {e}", config.backend_name),
    }
}

#[rstest]
#[tokio::test]
async fn test_chat_with_web_search_openai() {
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("test test_chat_with_web_search ... ignored, OPENAI_API_KEY not set");
            return;
        }
    };
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("gpt-5-nano")
        .max_tokens(5000)
        // .temperature(0.7)
        .openai_enable_web_search(true)
        .build()
        .expect("Failed to build LLM");

    match llm
        .chat_with_web_search("What is the weather in Tokyo?".to_string())
        .await
    {
        Ok(response) => {
            // println!("Response: {:?}", response.text());
            assert!(
                response.text().is_some() && !response.text().unwrap().is_empty(),
                "Expected response message, got {:?}",
                response.text()
            );
        }
        Err(e) => panic!("Chat error for OpenAI web search: {e}"),
    }
}

#[rstest]
#[case::openai(&BACKEND_CONFIGS[0])]
#[case::mistral(&BACKEND_CONFIGS[1])]
// #[case::google(&BACKEND_CONFIGS[2])]
#[case::groq(&BACKEND_CONFIGS[3])]
#[case::anthropic(&BACKEND_CONFIGS[5])]
#[case::openrouter(&BACKEND_CONFIGS[6])]
#[case::xai(&BACKEND_CONFIGS[7])]
#[tokio::test]
async fn test_list_models(#[case] config: &BackendTestConfig) {
    let api_key = match std::env::var(config.env_key) {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_{}_list_models ... ignored, {} not set",
                config.backend_name, config.env_key
            );
            return;
        }
    };

    let llm = LLMBuilder::new()
        .backend(config.backend.clone())
        .api_key(api_key)
        .build()
        .expect("Failed to build LLM");

    let request: Option<&ModelListRequest> = None;

    match llm.list_models(request).await {
        Ok(response) => {
            // Verify that we get a non-empty list of models
            let models = response.get_models();
            assert!(
                !models.is_empty(),
                "Expected at least one model, got empty list for {}",
                config.backend_name
            );

            // Verify that the backend matches what we expect
            assert_eq!(
                response.get_backend(),
                config.backend,
                "Expected backend {:?}, got {:?} for {}",
                config.backend,
                response.get_backend(),
                config.backend_name
            );

            // Verify that model IDs are non-empty strings
            for model_id in &models {
                assert!(
                    !model_id.is_empty(),
                    "Expected non-empty model ID, got empty string for {}",
                    config.backend_name
                );
            }

            // Log the number of models found for debugging
            println!(
                "Found {} models for {} backend",
                models.len(),
                config.backend_name
            );
        }
        Err(e) => panic!("List models error for {}: {e}", config.backend_name),
    }
}

#[tokio::test]
async fn test_anthropic_chat_stream_with_tools() {
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_anthropic_chat_stream_with_tools ... ignored, ANTHROPIC_API_KEY not set"
            );
            return;
        }
    };

    use llm::chat::StreamChunk;

    let llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(api_key)
        .model("claude-3-5-haiku-20241022")
        .max_tokens(512)
        .temperature(0.0) // Deterministic for testing
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get the current weather in a given location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city to get weather for"),
                )
                .required(vec!["location".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user()
        .content("What's the weather in Tokyo? Use the get_weather tool to find out.")
        .build()];

    match llm.chat_stream_with_tools(&messages, llm.tools()).await {
        Ok(mut stream) => {
            let mut text_chunks = Vec::new();
            let mut tool_starts = Vec::new();
            let mut tool_completes = Vec::new();
            let mut stop_reason = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        println!("Stream chunk: {:?}", chunk);
                        match chunk {
                            StreamChunk::Text(t) => text_chunks.push(t),
                            StreamChunk::ToolUseStart { id, name, .. } => {
                                tool_starts.push((id, name));
                            }
                            StreamChunk::ToolUseComplete { tool_call, .. } => {
                                tool_completes.push(tool_call);
                            }
                            StreamChunk::Done { stop_reason: sr } => {
                                stop_reason = Some(sr);
                            }
                            StreamChunk::ToolUseInputDelta { .. } => {
                                // These are intermediate chunks, we don't need to collect them
                            }
                        }
                    }
                    Err(e) => panic!("Stream error: {e}"),
                }
            }

            // Verify we got a tool call
            assert!(
                !tool_completes.is_empty(),
                "Expected at least one tool call, got none"
            );
            assert_eq!(
                tool_completes[0].function.name, "get_weather",
                "Expected get_weather tool call"
            );

            // Verify stop reason is tool_use
            assert_eq!(
                stop_reason,
                Some("tool_use".to_string()),
                "Expected stop_reason to be 'tool_use'"
            );

            // Verify the tool input contains location
            let args: serde_json::Value =
                serde_json::from_str(&tool_completes[0].function.arguments)
                    .expect("Failed to parse tool arguments as JSON");
            assert!(
                args.get("location").is_some(),
                "Expected 'location' in tool arguments"
            );
            println!(
                "Tool call arguments: {}",
                tool_completes[0].function.arguments
            );
        }
        Err(e) => panic!("Stream error: {e}"),
    }
}

#[tokio::test]
async fn test_anthropic_chat_stream_with_tools_text_only() {
    // Test that when tools are available but not needed, we still get text
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("test test_anthropic_chat_stream_with_tools_text_only ... ignored, ANTHROPIC_API_KEY not set");
            return;
        }
    };

    use llm::chat::StreamChunk;

    let llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(api_key)
        .model("claude-3-5-haiku-20241022")
        .max_tokens(512)
        .temperature(0.0)
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get the current weather in a given location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city to get weather for"),
                )
                .required(vec!["location".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    let messages = vec![ChatMessage::user()
        .content("Say hello in exactly 3 words.")
        .build()];

    match llm.chat_stream_with_tools(&messages, llm.tools()).await {
        Ok(mut stream) => {
            let mut text_chunks = Vec::new();
            let mut tool_completes = Vec::new();
            let mut stop_reason = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => match chunk {
                        StreamChunk::Text(t) => text_chunks.push(t),
                        StreamChunk::ToolUseComplete { tool_call, .. } => {
                            tool_completes.push(tool_call);
                        }
                        StreamChunk::Done { stop_reason: sr } => {
                            stop_reason = Some(sr);
                        }
                        _ => {}
                    },
                    Err(e) => panic!("Stream error: {e}"),
                }
            }

            // Verify we got text
            let complete_text: String = text_chunks.join("");
            assert!(
                !complete_text.is_empty(),
                "Expected text response, got empty"
            );
            println!("Text response: {}", complete_text);

            // Verify no tool calls
            assert!(
                tool_completes.is_empty(),
                "Expected no tool calls for this prompt"
            );

            // Verify stop reason is end_turn (not tool_use)
            assert_eq!(
                stop_reason,
                Some("end_turn".to_string()),
                "Expected stop_reason to be 'end_turn'"
            );
        }
        Err(e) => panic!("Stream error: {e}"),
    }
}

#[tokio::test]
async fn test_anthropic_resilient_chat_stream_with_tools() {
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!(
                "test test_anthropic_resilient_chat_stream_with_tools ... ignored, ANTHROPIC_API_KEY not set"
            );
            return;
        }
    };

    use llm::chat::StreamChunk;

    let llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(api_key)
        .model("claude-3-5-haiku-20241022")
        .max_tokens(512)
        .temperature(0.0)
        .resilient(true)
        .resilient_attempts(3)
        .resilient_backoff(200, 2_000)
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get the current weather in a given location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city to get weather for"),
                )
                .required(vec!["location".to_string()]),
        )
        .build()
        .expect("Failed to build resilient LLM");

    let messages = vec![ChatMessage::user()
        .content("What's the weather in Paris? Use the get_weather tool.")
        .build()];

    match llm.chat_stream_with_tools(&messages, llm.tools()).await {
        Ok(mut stream) => {
            let mut text_chunks = Vec::new();
            let mut tool_completes = Vec::new();
            let mut stop_reason = None;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        println!("Resilient stream chunk: {:?}", chunk);
                        match chunk {
                            StreamChunk::Text(t) => text_chunks.push(t),
                            StreamChunk::ToolUseComplete { tool_call, .. } => {
                                tool_completes.push(tool_call);
                            }
                            StreamChunk::Done { stop_reason: sr } => {
                                stop_reason = Some(sr);
                            }
                            _ => {}
                        }
                    }
                    Err(e) => panic!("Resilient stream error: {e}"),
                }
            }

            // Verify we got a tool call
            assert!(
                !tool_completes.is_empty(),
                "Expected at least one tool call with resilient wrapper"
            );
            assert_eq!(
                tool_completes[0].function.name, "get_weather",
                "Expected get_weather tool call"
            );

            // Verify stop reason
            assert_eq!(
                stop_reason,
                Some("tool_use".to_string()),
                "Expected stop_reason to be 'tool_use'"
            );

            println!(
                "Resilient streaming with tools succeeded. Tool args: {}",
                tool_completes[0].function.arguments
            );
        }
        Err(e) => panic!("Failed to start resilient stream: {e}"),
    }
}

// Note: test_anthropic_resilient_chat_stream_struct removed because
// Anthropic backend does not implement chat_stream_struct

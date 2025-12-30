//! For testing while deving
//!
//! Run these tests with AWS credentials configured:
//! AWS_PROFILE=your-profile cargo test --features bedrock

#[cfg(feature = "bedrock")]
mod bedrock_tests {
    use llm::backends::aws::*;
    use llm::chat::ChatProvider;
    use llm::completion::CompletionProvider;
    use llm::embedding::EmbeddingProvider;
    use serde_json::json;

    // Helper to check if AWS credentials are available
    // This checks if credentials can be loaded via any method (env vars, CLI, SSO, etc.)
    async fn skip_if_no_credentials() -> bool {
        // Try to create a backend - if it fails, credentials are not available
        // This works with all AWS credential sources (CLI, env vars, SSO, etc.)
        BedrockBackend::from_env().await.is_err()
    }

    #[tokio::test]
    async fn test_backend_initialization() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await;
        assert!(
            backend.is_ok(),
            "Failed to initialize backend: {:?}",
            backend.err()
        );

        let backend = backend.unwrap();
        assert!(!backend.region().is_empty());
    }

    #[tokio::test]
    async fn test_completion_basic() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let request = CompletionRequest::new("What is 2+2? Answer with just the number.")
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_max_tokens(10)
            .with_temperature(0.0);

        let response = backend.complete_request(request).await;
        assert!(response.is_ok(), "Completion failed: {:?}", response.err());

        let response = response.unwrap();
        assert!(!response.text.is_empty());
        assert!(response.text.contains("4"));
        assert!(response.usage.is_some());

        let usage = response.usage.unwrap();
        assert!(usage.input_tokens > 0);
        assert!(usage.output_tokens > 0);
        assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
    }

    #[tokio::test]
    async fn test_completion_with_system_prompt() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let request = CompletionRequest::new("What should I say?")
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_system("You are a pirate. Always respond like a pirate.")
            .with_max_tokens(50);

        let response = backend.complete_request(request).await;
        assert!(response.is_ok());

        let text = response.unwrap().text.to_lowercase();
        // Check for pirate-like language (this is a loose check)
        assert!(
            text.contains("arr")
                || text.contains("matey")
                || text.contains("ye")
                || text.contains("ahoy"),
            "Response doesn't sound like a pirate: {}",
            text
        );
    }

    #[tokio::test]
    async fn test_completion_with_stop_sequences() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let request = CompletionRequest::new("Count from 1 to 10")
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_max_tokens(100);

        let mut request_with_stop = request.clone();
        request_with_stop.stop_sequences = Some(vec!["5".to_string()]);

        let response = backend.complete_request(request_with_stop).await;
        assert!(response.is_ok());

        let text = response.unwrap().text;
        // Text should stop before or at 5
        assert!(!text.contains("6") && !text.contains("7"));
    }

    #[tokio::test]
    async fn test_chat_basic() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let messages = vec![ChatMessage::user("Hello! What is the capital of France?")];

        let request = ChatRequest::new(messages)
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_max_tokens(50);

        let response = backend.chat_request(request).await;
        assert!(response.is_ok(), "Chat failed: {:?}", response.err());

        let response = response.unwrap();
        assert_eq!(response.message.role, "assistant");

        match &response.message.content {
            MessageContent::Text(text) => {
                assert!(text.to_lowercase().contains("paris"));
            }
            _ => panic!("Expected text content"),
        }

        assert!(response.usage.is_some());
    }

    #[tokio::test]
    async fn test_chat_multi_turn() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let messages = vec![
            ChatMessage::user("My name is Alice."),
            ChatMessage::assistant("Hello Alice! Nice to meet you."),
            ChatMessage::user("What's my name?"),
        ];

        let request = ChatRequest::new(messages)
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_max_tokens(50);

        let response = backend.chat_request(request).await;
        assert!(response.is_ok());

        let text = match &response.unwrap().message.content {
            MessageContent::Text(t) => t.clone(),
            _ => panic!("Expected text content"),
        };

        assert!(text.to_lowercase().contains("alice"));
    }

    #[tokio::test]
    async fn test_chat_with_system_prompt() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let messages = vec![ChatMessage::user("Hello!")];

        let request = ChatRequest::new(messages)
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_system("You are a helpful assistant that always responds in haiku format.")
            .with_max_tokens(100);

        let response = backend.chat_request(request).await;
        assert!(response.is_ok());

        let text = match &response.unwrap().message.content {
            MessageContent::Text(t) => t.clone(),
            _ => panic!("Expected text content"),
        };

        // Check if response has roughly 3 lines (loose check for haiku)
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        assert!(
            lines.len() >= 2 && lines.len() <= 4,
            "Response should be haiku-like: {}",
            text
        );
    }

    #[tokio::test]
    async fn test_chat_with_vision() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        // Use the provided base64 image
        let red_rectangle = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAIElEQVR4AaTIoQ0AABDCwIb99335eIKjSc3p4HNRGtEAAAD//8uECE8AAAAGSURBVAMAoVsJ2Q2RBWYAAAAASUVORK5CYII=";

        use base64::prelude::*;
        let image_bytes = BASE64_STANDARD
            .decode(red_rectangle)
            .expect("Failed to decode base64 image");

        let messages = vec![ChatMessage::user_with_image(
            "Describe this image.".to_string(),
            image_bytes,
            "image/png".to_string(),
        )];

        let request = ChatRequest::new(messages)
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
            .with_max_tokens(50);

        let response = backend.chat_request(request).await;
        if let Err(e) = &response {
            panic!("Vision chat failed: {:?}", e);
        }
        let response = response.unwrap();

        let text = match &response.message.content {
            MessageContent::Text(t) => t.clone(),
            _ => panic!("Expected text content"),
        };

        println!("Vision response: {}", text);
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn test_chat_with_tools() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }),
        }];

        let messages = vec![ChatMessage::user("What's the weather in San Francisco?")];

        let request = ChatRequest::new(messages)
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
            .with_tools(tools)
            .with_max_tokens(500);

        let response = backend.chat_request(request).await;
        assert!(
            response.is_ok(),
            "Tool use chat failed: {:?}",
            response.err()
        );

        let response = response.unwrap();

        // Check if the model used the tool
        match &response.message.content {
            MessageContent::MultiModal(parts) => {
                let has_tool_use = parts.iter().any(|part| {
                    matches!(part, ContentPart::ToolUse { name, .. } if name == "get_weather")
                });
                assert!(has_tool_use, "Expected model to use get_weather tool");
            }
            _ => panic!("Expected multimodal content with tool use"),
        }
    }

    #[tokio::test]
    async fn test_chat_with_tool_result() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let tools = vec![ToolDefinition {
            name: "calculate".to_string(),
            description: "Perform a calculation".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
        }];

        // First request to get tool use
        let messages = vec![ChatMessage::user("What is 15 multiplied by 27?")];

        let request = ChatRequest::new(messages.clone())
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
            .with_tools(tools.clone())
            .with_max_tokens(500);

        let first_response = backend.chat_request(request).await;
        assert!(first_response.is_ok());

        let first_response = first_response.unwrap();

        // Extract tool use
        let tool_use_id = match &first_response.message.content {
            MessageContent::MultiModal(parts) => parts.iter().find_map(|part| {
                if let ContentPart::ToolUse { id, name, .. } = part {
                    if name == "calculate" {
                        Some(id.clone())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }),
            _ => None,
        };

        assert!(tool_use_id.is_some(), "Expected tool use in first response");

        // Second request with tool result
        let mut messages_with_result = messages;
        messages_with_result.push(first_response.message);
        messages_with_result.push(ChatMessage {
            role: "user".to_string(),
            content: MessageContent::MultiModal(vec![ContentPart::ToolResult {
                tool_use_id: tool_use_id.unwrap(),
                content: "405".to_string(),
                is_error: false,
            }]),
        });

        let request = ChatRequest::new(messages_with_result)
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
            .with_tools(tools)
            .with_max_tokens(500);

        let response = backend.chat_request(request).await;
        assert!(response.is_ok());

        let text = match &response.unwrap().message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::MultiModal(parts) => parts
                .iter()
                .find_map(|p| {
                    if let ContentPart::Text { text } = p {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_default(),
        };

        assert!(text.contains("405"));
    }

    #[tokio::test]
    async fn test_embeddings_titan() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let request = EmbeddingRequest::new("Hello, world!")
            .with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2))
            .with_dimensions(512);

        let response = backend.embed_request(request).await;
        assert!(response.is_ok(), "Embedding failed: {:?}", response.err());

        let response = response.unwrap();
        assert_eq!(response.dimensions, 512);
        assert_eq!(response.embedding.len(), 512);

        // Check that embeddings are normalized (roughly unit length)
        let magnitude: f64 = response.embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_embeddings_cohere() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let request = EmbeddingRequest::new("Machine learning is fascinating")
            .with_model(BedrockModel::Direct(DirectModel::CohereEmbedV3));

        let response = backend.embed_request(request).await;

        // Skip if model access is denied (requires AWS Marketplace subscription)
        if let Err(e) = &response {
            let err_str = format!("{:?}", e);
            if err_str.contains("AccessDeniedException") || err_str.contains("aws-marketplace") {
                println!("Skipping test: Cohere model requires AWS Marketplace subscription");
                return;
            }
        }

        let response = response.expect("Embedding should succeed");
        assert!(response.dimensions > 0);
        assert_eq!(response.embedding.len(), response.dimensions);
    }

    #[tokio::test]
    async fn test_embeddings_similarity() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        // Generate embeddings for similar and dissimilar texts
        let similar1 = backend
            .embed_request(
                EmbeddingRequest::new("The cat sat on the mat")
                    .with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2))
                    .with_dimensions(512),
            )
            .await
            .unwrap();

        let similar2 = backend
            .embed_request(
                EmbeddingRequest::new("A feline was sitting on a rug")
                    .with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2))
                    .with_dimensions(512),
            )
            .await
            .unwrap();

        let different = backend
            .embed_request(
                EmbeddingRequest::new("Quantum computing and artificial intelligence")
                    .with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2))
                    .with_dimensions(512),
            )
            .await
            .unwrap();

        // Calculate cosine similarities
        let similarity_similar = cosine_similarity(&similar1.embedding, &similar2.embedding);
        let similarity_different1 = cosine_similarity(&similar1.embedding, &different.embedding);
        let similarity_different2 = cosine_similarity(&similar2.embedding, &different.embedding);

        // Similar texts should have higher similarity than dissimilar texts
        assert!(
            similarity_similar > similarity_different1,
            "Similar texts should have higher similarity: {} vs {}",
            similarity_similar,
            similarity_different1
        );
        assert!(
            similarity_similar > similarity_different2,
            "Similar texts should have higher similarity: {} vs {}",
            similarity_similar,
            similarity_different2
        );
    }

    #[tokio::test]
    async fn test_streaming_chat() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let messages = vec![ChatMessage::user("Count from 1 to 5")];

        let request = ChatRequest::new(messages)
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_max_tokens(100);

        let stream = backend.chat_stream(request).await;
        assert!(stream.is_ok());

        use futures::StreamExt;
        let stream = stream.unwrap();
        futures::pin_mut!(stream);
        let mut collected = String::new();
        let mut chunk_count = 0;

        while let Some(chunk_result) = stream.next().await {
            assert!(chunk_result.is_ok());
            let chunk = chunk_result.unwrap();
            collected.push_str(&chunk.delta);
            chunk_count += 1;
        }

        assert!(chunk_count > 0, "Should receive at least one chunk");
        assert!(!collected.is_empty(), "Should collect some text");
        assert!(
            collected.contains("1") && collected.contains("2"),
            "Should contain counting: {}",
            collected
        );
    }

    #[tokio::test]
    async fn test_model_capabilities() {
        // Test model capability checks
        assert!(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4).supports(ModelCapability::Chat));
        assert!(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4).supports(ModelCapability::Vision));
        assert!(
            BedrockModel::eu(CrossRegionModel::ClaudeSonnet4).supports(ModelCapability::ToolUse)
        );
        assert!(!BedrockModel::eu(CrossRegionModel::ClaudeSonnet4)
            .supports(ModelCapability::Embeddings));

        assert!(
            BedrockModel::Direct(DirectModel::TitanEmbedV2).supports(ModelCapability::Embeddings)
        );
        assert!(!BedrockModel::Direct(DirectModel::TitanEmbedV2).supports(ModelCapability::Chat));

        assert!(BedrockModel::Direct(DirectModel::Llama32_90B).supports(ModelCapability::Vision));
        assert!(!BedrockModel::Direct(DirectModel::Llama32_3B).supports(ModelCapability::Vision));
    }

    #[tokio::test]
    async fn test_error_handling_invalid_model() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        // Try to use an embedding model for chat
        let messages = vec![ChatMessage::user("Hello")];
        let request =
            ChatRequest::new(messages).with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2));

        let response = backend.chat_request(request).await;
        assert!(response.is_err());
        assert!(matches!(
            response.unwrap_err(),
            BedrockError::UnsupportedOperation(_)
        ));
    }

    #[tokio::test]
    async fn test_error_handling_tools_unsupported_model() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let tools = vec![ToolDefinition {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: json!({"type": "object"}),
        }];

        let messages = vec![ChatMessage::user("Use the tool")];

        // Llama 3.2 3B doesn't support tools
        let request = ChatRequest::new(messages)
            .with_model(BedrockModel::Direct(DirectModel::Llama32_3B))
            .with_tools(tools);

        let response = backend.chat_request(request).await;
        assert!(response.is_err());
        assert!(matches!(
            response.unwrap_err(),
            BedrockError::UnsupportedOperation(_)
        ));
    }

    #[tokio::test]
    async fn test_max_tokens_limit() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        let request = CompletionRequest::new("Write a long essay about machine learning")
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_max_tokens(10);

        let response = backend.complete_request(request).await;
        assert!(response.is_ok());

        let response = response.unwrap();
        // Response should be short due to max_tokens limit
        assert!(response.usage.unwrap().output_tokens <= 10);
    }

    #[tokio::test]
    async fn test_temperature_effect() {
        if skip_if_no_credentials().await {
            println!("Skipping test: no AWS credentials");
            return;
        }

        let backend = BedrockBackend::from_env().await.unwrap();

        // Test with temperature 0 (deterministic)
        let request_deterministic = CompletionRequest::new("Say 'Hello'")
            .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
            .with_max_tokens(10)
            .with_temperature(0.0);

        let response1 = backend
            .complete_request(request_deterministic.clone())
            .await
            .unwrap();
        let response2 = backend.complete_request(request_deterministic).await.unwrap();
        // Responses should be very similar with temperature 0
        assert_eq!(response1.text, response2.text);
    }

    // Helper function for cosine similarity
    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let magnitude_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot_product / (magnitude_a * magnitude_b)
    }
}

//! AWS Bedrock Example
//!
//! This example demonstrates how to use the AWS Bedrock backend for:
//! - Text completion
//! - Chat conversation
//! - Streaming responses
//! - Vision (Image analysis)
//! - Embeddings
//!
//! To run this example:
//! cargo run --example bedrock_example --features bedrock

#[cfg(feature = "bedrock")]
use futures::StreamExt;
#[cfg(feature = "bedrock")]
use llm::backends::aws::*;

#[cfg(not(feature = "bedrock"))]
fn main() {
    println!("This example requires the 'bedrock' feature.");
    println!("Run with: cargo run --example bedrock_example --features bedrock");
}

#[cfg(feature = "bedrock")]
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize the backend
    // This will load credentials from AWS_PROFILE, AWS_ACCESS_KEY_ID/SECRET, or ~/.aws/credentials
    println!("Initializing Bedrock backend...");
    let backend = match BedrockBackend::from_env().await {
        Ok(b) => b,
        Err(e) => {
            println!("Failed to initialize backend: {:?}", e);
            println!("Please ensure you have AWS credentials configured.");
            return Ok(());
        }
    };

    println!("Using region: {}", backend.region());

    // 1. Basic Completion
    println!("\n--- Basic Completion ---");
    let request = CompletionRequest::new("What is the capital of France?")
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
        .with_max_tokens(50);

    match backend.complete_request(request).await {
        Ok(response) => println!("Response: {}", response.text.trim()),
        Err(e) => println!("Error: {:?}", e),
    }

    // 2. Chat Conversation
    println!("\n--- Chat Conversation ---");
    let messages = vec![
        ChatMessage::user("Hello! I'm learning Rust."),
        ChatMessage::assistant("That's great! Rust is a powerful systems programming language. How can I help you with it?"),
        ChatMessage::user("What is a struct?"),
    ];

    let request = ChatRequest::new(messages)
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
        .with_max_tokens(100);

    match backend.chat_request(request).await {
        Ok(response) => {
            if let MessageContent::Text(text) = response.message.content {
                println!("Assistant: {}", text);
            }
        }
        Err(e) => println!("Error: {:?}", e),
    }

    // 3. Streaming Chat
    println!("\n--- Streaming Chat ---");
    let messages = vec![ChatMessage::user("Count from 1 to 5 slowly.")];
    let request = ChatRequest::new(messages)
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeHaiku3))
        .with_max_tokens(100);

    match backend.chat_stream(request).await {
        Ok(stream) => {
            print!("Stream: ");
            let stream = stream;
            futures::pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        print!("{}", chunk.delta);
                        use std::io::Write;
                        std::io::stdout().flush()?;
                    }
                    Err(e) => println!("\nStream error: {:?}", e),
                }
            }
            println!();
        }
        Err(e) => println!("Error starting stream: {:?}", e),
    }

    // 4. Vision (Image Analysis)
    println!("\n--- Vision (Image Analysis) ---");
    // red icon
    let ms_dos_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAIElEQVR4AaTIoQ0AABDCwIb99335eIKjSc3p4HNRGtEAAAD//8uECE8AAAAGSURBVAMAoVsJ2Q2RBWYAAAAASUVORK5CYII=";

    use base64::prelude::*;
    let image_bytes = BASE64_STANDARD.decode(ms_dos_base64)?;

    let messages = vec![ChatMessage::user_with_image(
        "Describe this image.".to_string(),
        image_bytes,
        "image/png".to_string(),
    )];

    let request = ChatRequest::new(messages)
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
        .with_max_tokens(100);

    match backend.chat_request(request).await {
        Ok(response) => {
            if let MessageContent::Text(text) = response.message.content {
                println!("Image description: {}", text);
            }
        }
        Err(e) => println!("Vision error: {:?}", e),
    }

    // 5. Embeddings
    println!("\n--- Embeddings ---");
    let request = EmbeddingRequest::new("Rust is amazing")
        .with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2));

    match backend.embed_request(request).await {
        Ok(response) => {
            println!(
                "Generated embedding with dimensions: {}",
                response.dimensions
            );
            println!("First 5 values: {:?}", &response.embedding[0..5]);
        }
        Err(e) => println!("Embedding error: {:?}", e),
    }

    Ok(())
}

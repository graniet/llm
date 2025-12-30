# Quick Start Guide - AWS Bedrock Backend

Get up and running with the AWS Bedrock backend in 5 minutes!

## Prerequisites

- Rust 1.70 or later
- AWS account with Bedrock access
- AWS credentials configured

## Step 1: Add Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
fork-llm = { version = "0.1.0", features = ["bedrock"] }
tokio = { version = "1.0", features = ["full"] }
```

## Step 2: Configure AWS Credentials

Choose one method:

### Option A: Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=us-east-1
```

### Option B: AWS Profile
```bash
export AWS_PROFILE=your-profile
```

### Option C: AWS Config File
Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_key
aws_secret_access_key = your_secret
region = us-east-1
```

## Step 3: Write Your First Program

```rust
use fork_llm::backends::bedrock::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize backend
    let backend = BedrockBackend::new().await?;
    
    // Ask a question
    let request = CompletionRequest::new("What is Rust?")
        .with_model(BedrockModel::ClaudeHaiku3)
        .with_max_tokens(100);
    
    let response = backend.complete(request).await?;
    println!("{}", response.text);
    
    Ok(())
}
```

## Step 4: Run It

```bash
cargo run
```

## Common Use Cases

### Model Selection
```rust
// Cross region models
BedrockModel::eu(CrossRegionModel::ClaudeSonnet4)
BedrockModel::us(CrossRegionModel::ClaudeSonnet4)

// Cross region with more control for missing regions
BedrockModel::cross_region("eu-central-1", CrossRegionModel::ClaudeSonnet4)

// Direct models
BedrockModel::Direct(DirectModel::ClaudeSonnet4)

// From ARN or ID
BedrockModel::from_id("arn:aws:bedrock:eu-central-1:...")
BedrockModel::from_id("us.anthropic.claude-sonnet-4-0-v1:0")
```

### Model Verification
```rust
let model = BedrockModel::eu(CrossRegionModel::ClaudeSonnet4);

// Get ARN or model ID
model.model_id()  // Returns full ARN

// Check if cross-region
model.is_cross_region_profile()  // Returns true

// Check capabilities
model.supports(ModelCapability::Chat)       // true
model.supports(ModelCapability::Vision)     // true
model.supports(ModelCapability::ToolUse)    // true
model.supports(ModelCapability::Embeddings) // false

// Get limits
model.context_window()
model.max_output_tokens()
```

### Simple Q&A
```rust
let response = backend.complete(
    CompletionRequest::new("What is 2+2?")
        .with_max_tokens(10)
).await?;
```

### Chat Conversation
```rust
let messages = vec![
    ChatMessage::user("Hello!"),
];

let response = backend.chat(
    ChatRequest::new(messages)
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
).await?;
```

### Image Analysis
```rust
let image = std::fs::read("photo.jpg")?;

let messages = vec![
    ChatMessage::user_with_image(
        "What's in this image?".to_string(),
        image,
        "image/jpeg".to_string(),
    ),
];

let response = backend.chat(
    ChatRequest::new(messages)
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
).await?;
```

### Generate Embeddings
```rust
let response = backend.embed(
    EmbeddingRequest::new("Hello, world!")
        .with_model(BedrockModel::Direct(DirectModel::TitanEmbedV2))
).await?;

println!("Embedding: {:?}", response.embedding);
```

## Troubleshooting

### Error: No credentials found
**Solution**: Set AWS credentials (see Step 2)

### Error: ResourceNotFoundException
**Solution**: Model not available in your region. Try:
- Use `us-east-1` region
- Check model availability: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html

### Error: AccessDeniedException
**Solution**: Add Bedrock permissions to your IAM user/role:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ],
    "Resource": "*"
  }]
}
```

## Example Projects

### Chatbot
```rust
let mut conversation = Vec::new();

loop {
    let user_input = read_user_input();
    conversation.push(ChatMessage::user(user_input));
    
    let response = backend.chat(
        ChatRequest::new(conversation.clone())
    ).await?;
    
    conversation.push(response.message.clone());
    
    // Display response
    match response.message.content {
        MessageContent::Text(text) => println!("Bot: {}", text),
        _ => {}
    }
}
```

### Document Q&A
```rust
// Generate embeddings for documents
let doc_embeddings = vec![];
for doc in documents {
    let emb = backend.embed(
        EmbeddingRequest::new(doc)
    ).await?;
    doc_embeddings.push(emb.embedding);
}

// Find relevant docs for query
let query_emb = backend.embed(
    EmbeddingRequest::new(query)
).await?;

let relevant_docs = find_similar(query_emb, doc_embeddings);

// Answer using relevant context
let messages = vec![
    ChatMessage::user(format!(
        "Context: {}\n\nQuestion: {}",
        relevant_docs, query
    )),
];

let answer = backend.chat(
    ChatRequest::new(messages)
).await?;
```

### Tool-Using Agent
```rust
let tools = vec![
    ToolDefinition {
        name: "search".to_string(),
        description: "Search the web".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }),
    },
];

let response = backend.chat(
    ChatRequest::new(messages)
        .with_tools(tools)
).await?;

// Handle tool use and continue conversation
```

## Resources

- 📚 [Full Documentation](README.md)
- 🔧 [Integration Guide](INTEGRATION.md)
- 💻 [Examples](examples/bedrock_demo.rs)
- 🧪 [Tests](tests/bedrock_tests.rs)
- 🌐 [AWS Bedrock Docs](https://docs.aws.amazon.com/bedrock/)

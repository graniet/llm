# Quick Start Guide - AWS Bedrock Backend

## Prerequisites

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

### Option D: AWS CLI

Login with aws cli, then the SDK will pick up those credentials automatically.

## Step 3: Use it

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

Usually, however, you would use the builder:

```rust
let mut llm = LLMBuilder::new()
    .backend(LLMBackend::Bedrock)
    .model(BedrockModel::ClaudeHaiku3)
    .base_url(region)  // Optional, defaults to AWS_REGION
    .build();

let response = llm.complete("What is Rust?").await?;
println!("{}", response.text);
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

### External Capability Overrides
If you need to use models not baked into the crate (or disable tool use on a specific model),
provide overrides via environment variables.

```bash
export LLM_BEDROCK_MODEL_CAPABILITIES_PATH=/path/to/bedrock_capabilities.toml
```

Example TOML:
```toml
[[model]]
name = "arn:aws:bedrock:eu-central-1:876164100382:inference-profile/eu.anthropic.claude-sonnet-4-20250514-v1:0"
completion = true
chat = true
embeddings = false
vision = true
tool_use = true
streaming = true
```

Inline config also works:
```bash
export LLM_BEDROCK_MODEL_CAPABILITIES=$'[[model]]\nname="arn:aws:bedrock:eu-central-1:876164100382:inference-profile/eu.anthropic.claude-sonnet-4-20250514-v1:0"\ncompletion=true\nchat=true\nembeddings=false\nvision=true\ntool_use=true\nstreaming=true'
```

Each entry requires `name` (model ID or ARN). Supported keys per model: `completion`, `chat`, `embeddings`, `vision`, `tool_use`, `streaming`.

## Troubleshooting

### Error: No credentials found
**Solution**: Set AWS credentials (see Step 2)

### Error: ResourceNotFoundException
**Solution**: Model not available in your region. Try:
- Use `us-east-1` region
- Check model availability: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html

### Error: AccessDeniedException
**Solution**: Add Bedrock permissions to your IAM user/role, or the policy the infra is using:
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
# LLM

> **Note**: This crate name previously belonged to another project. The current implementation represents a new and different library. The previous crate is now archived and will not receive any updates. **ref: https://github.com/rustformers/llm**

**LLM** is a **Rust** library that lets you use **multiple LLM backends** in a single project: [OpenAI](https://openai.com), [Anthropic (Claude)](https://www.anthropic.com), [Ollama](https://github.com/ollama/ollama), [DeepSeek](https://www.deepseek.com), [xAI](https://x.ai), [Phind](https://www.phind.com), [Groq](https://www.groq.com) and [Google](https://cloud.google.com/gemini).
With a **unified API** and **builder style** - similar to the Stripe experience - you can easily create **chat** or text **completion** requests without multiplying structures and crates.

## Key Features

- **Multi-backend**: Manage OpenAI, Anthropic, Ollama, DeepSeek, xAI, Phind, Groq and Google through a single entry point.
- **Multi-step chains**: Create multi-step chains with different backends at each step.
- **Templates**: Use templates to create complex prompts with variables.
- **Builder pattern**: Configure your LLM (model, temperature, max_tokens, timeouts...) with a few simple calls.
- **Chat & Completions**: Two unified traits (`ChatProvider` and `CompletionProvider`) to cover most use cases.
- **Extensible**: Easily add new backends.
- **Rust-friendly**: Designed with clear traits, unified error handling, and conditional compilation via *features*.
- **Validation**: Add validation to your requests to ensure the output is what you expect.
- **Evaluation**: Add evaluation to your requests to score the output of LLMs.
- **Parallel Evaluation**: Evaluate multiple LLM providers in parallel and select the best response based on scoring functions.
- **Function calling**: Add function calling to your requests to use tools in your LLMs.
- **REST API**: Serve any LLM backend as a REST API with openai standard format.
- **Vision**: Add vision to your requests to use images in your LLMs.
- **Reasoning**: Add reasoning to your requests to use reasoning in your LLMs.

## Use any LLM backend on your project

Simply add **LLM** to your `Cargo.toml`:

```toml
[dependencies]
llm = { version = "1.0.4", features = ["openai", "anthropic", "ollama", "deepseek", "xai", "phind", "google", "groq"] }
```

## Use any LLM on cli

LLM includes a command-line tool for easily interacting with different LLM models. You can install it with: ```cargo install llm```

- Use `llm` to start an interactive chat session
- Use `llm openai:gpt-4o` to start an interactive chat session with provider:model
- Use `llm set OPENAI_API_KEY your_key` to configure your API key
- Use `llm default openai:gpt-4` to set a default provider
- Use `echo "Hello World" | llm` to pipe
- Use `llm --provider openai --model gpt-4 --temperature 0.7` for advanced options

## Multi-step LLM Chains

LLM also provides a CLI tool for creating and running multi-step prompt chains:

```bash
# Create a chain template
llm-chain create --output=my-chain.yaml

# View available providers
llm-chain providers

# Run a chain with specific provider
llm-chain --file=my-chain.yaml --provider=openai:gpt-4o
```

Chain definitions are YAML or JSON files that specify a sequence of prompts with variable substitution and conditional execution:

```yaml
name: example-chain
description: A chain that demonstrates multi-step processing with conditionals
default_provider: openai:gpt-4o
input_var: input  # Variable name for piped input
steps:
  - id: topic
    template: Suggest an interesting technical topic to explore based on {{input}}. Answer with just the topic name.
    temperature: 0.7
    max_tokens: 50
  - id: details
    template: List 3 key aspects of {{topic}} that developers should know.
    temperature: 0.5
    max_tokens: 200
  - id: library_check
    template: Is there a popular library for {{topic}}? Answer with library name or 'none'.
    temperature: 0.3
    max_tokens: 50
  - id: library_details
    template: Describe the key features of the {{library_check}} library.
    temperature: 0.3
    max_tokens: 200
    condition: "!library_check=none"  # Only run if library_check is not 'none'
  - id: code_example
    template: 'Based on {{topic}} and these aspects: {{details}}, provide a code example.'
    temperature: 0.3
    max_tokens: 400
  - id: system_info
    template: This analysis was generated on {{sys.date}} at {{sys.time}} on a {{sys.os}} system.
    temperature: 0.1
    max_tokens: 50
```

Variables from previous steps can be referenced using `{{variable_name}}` syntax. The chain also includes system variables (like `{{sys.date}}`, `{{sys.time}}`, `{{sys.os}}`) and supports conditional step execution. Interactive mode is also available for steps that require human review:

```bash
# Run in fully interactive mode
llm-chain --file=my-chain.yaml --interactive

# Make only specific steps interactive
llm-chain --file=my-chain.yaml --interactive-steps=topic,code_example

# Save interaction history for later review or replay
llm-chain --file=my-chain.yaml --interactive --save-history=session.json
```

You can also pipe input to chain execution:

```bash
# Pipe input to the chain
echo "machine learning" | llm-chain --file=my-chain.yaml

# Use the final result in other commands
echo "functional programming" | llm-chain --file=my-chain.yaml | grep "function"

# Get the output in JSON format for programmatic use
llm-chain --file=my-chain.yaml --json
echo "design a mobile app" | llm-chain --file=my-chain.yaml --json > result.json
```

## Serving any LLM backend as a REST API
- Use standard messages format
- Use step chains to chain multiple LLM backends together
- Expose the chain through a REST API with openai standard format

```shell
[dependencies]
llm = { version = "1.0.4", features = ["openai", "anthropic", "ollama", "deepseek", "xai", "phind", "google", "groq", "api"] }
```

More details in the [`api_example`](examples/api_example.rs)

## More examples

| Name | Description |
|------|-------------|
| [`anthropic_example`](examples/anthropic_example.rs) | Demonstrates integration with Anthropic's Claude model for chat completion |
| [`chain_example`](examples/chain_example.rs) | Shows how to create multi-step prompt chains for exploring programming language features |
| [`deepseek_example`](examples/deepseek_example.rs) | Basic DeepSeek chat completion example with deepseek-chat models |
| [`embedding_example`](examples/embedding_example.rs) | Basic embedding example with OpenAI's API |
| [`multi_backend_example`](examples/multi_backend_example.rs) | Illustrates chaining multiple LLM backends (OpenAI, Anthropic, DeepSeek) together in a single workflow |
| [`ollama_example`](examples/ollama_example.rs) | Example of using local LLMs through Ollama integration |
| [`openai_example`](examples/openai_example.rs) | Basic OpenAI chat completion example with GPT models |
| [`phind_example`](examples/phind_example.rs) | Basic Phind chat completion example with Phind-70B model |
| [`validator_example`](examples/validator_example.rs) | Basic validator example with Anthropic's Claude model |
| [`xai_example`](examples/xai_example.rs) | Basic xAI chat completion example with Grok models |
| [`evaluation_example`](examples/evaluation_example.rs) | Basic evaluation example with Anthropic, Phind and DeepSeek |
| [`evaluator_parallel_example`](examples/evaluator_parallel_example.rs) | Evaluate multiple LLM providers in parallel |
| [`google_example`](examples/google_example.rs) | Basic Google Gemini chat completion example with Gemini models |
| [`google_pdf`](examples/google_pdf.rs) | Google Gemini chat with PDF attachment |
| [`google_image`](examples/google_image.rs) | Google Gemini chat with PDF attachment |
| [`google_embedding_example`](examples/google_embedding_example.rs) | Basic Google Gemini embedding example with Gemini models |
| [`tool_calling_example`](examples/tool_calling_example.rs) | Basic tool calling example with OpenAI |
| [`deepclaude_pipeline_example`](examples/deepclaude_pipeline_example.rs) | Basic deepclaude pipeline example with DeepSeek and Claude |
| [`api_example`](examples/api_example.rs) | Basic API (openai standard format) example with OpenAI, Anthropic, DeepSeek and Groq |
| [`api_deepclaude_example`](examples/api_deepclaude_example.rs) | Basic API (openai standard format) example with DeepSeek and Claude |
| [`anthropic_vision_example`](examples/anthropic_vision_example.rs) | Basic anthropic vision example with Anthropic |
| [`openai_vision_example`](examples/openai_vision_example.rs) | Basic openai vision example with OpenAI |
| [`openai_reasoning_example`](examples/openai_reasoning_example.rs) | Basic openai reasoning example with OpenAI |
| [`anthropic_thinking_example`](examples/anthropic_thinking_example.rs) | Anthropic reasoning example |

## Usage
Here's a basic example using OpenAI for chat completion. See the examples directory for other backends (Anthropic, Ollama, DeepSeek, xAI, Google, Phind), embedding capabilities, and more advanced use cases.

use clap::Parser;
use colored::*;
use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::{ChatMessage, ImageMime};
use llm::secret_store::SecretStore;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use spinners::{Spinner, Spinners};
use std::io::{self, IsTerminal, Read, Write};
use std::str::FromStr;

#[cfg(feature = "logging")]
use env_logger;

/// Command line arguments for the LLM CLI
#[derive(Parser)]
#[clap(
    name = "llm",
    about = "Interactive CLI interface for chatting with LLM providers",
    allow_hyphen_values = true
)]
struct CliArgs {
    /// Command to execute (chat, set, get, delete, default)
    #[arg(index = 1)]
    command: Option<String>,

    /// Provider string in format "provider:model" or secret key for set/get/delete commands
    #[arg(index = 2)]
    provider_or_key: Option<String>,

    /// Initial prompt or secret value for set command
    #[arg(index = 3)]
    prompt_or_value: Option<String>,

    /// LLM provider name
    #[arg(long)]
    provider: Option<String>,

    /// Model name to use
    #[arg(long)]
    model: Option<String>,

    /// System prompt to set context
    #[arg(long)]
    system: Option<String>,

    /// API key for the provider
    #[arg(long)]
    api_key: Option<String>,

    /// Base URL for the API
    #[arg(long)]
    base_url: Option<String>,

    /// Temperature setting (0.0-1.0)
    #[arg(long)]
    temperature: Option<f32>,

    /// Maximum tokens in the response
    #[arg(long)]
    max_tokens: Option<u32>,
}

/// Detects the MIME type of an image from its binary data
///
/// # Arguments
///
/// * `data` - The binary data of the image
///
/// # Returns
///
/// * `Some(ImageMime)` - The detected MIME type if recognized
/// * `None` - If the image format is not recognized
fn detect_image_mime(data: &[u8]) -> Option<ImageMime> {
    if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(ImageMime::JPEG)
    } else if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        Some(ImageMime::PNG)
    } else if data.starts_with(&[0x47, 0x49, 0x46]) {
        Some(ImageMime::GIF)
    } else {
        None
    }
}

/// Retrieves provider and model information from various sources
///
/// # Arguments
///
/// * `args` - Command line arguments
///
/// # Returns
///
/// * `Some((provider_name, model_name))` - Provider name and optional model name
/// * `None` - If no provider information could be found
fn get_provider_info(args: &CliArgs) -> Option<(String, Option<String>)> {
    if let Some(default_provider) = SecretStore::new()
        .ok()
        .and_then(|store| store.get_default_provider().cloned())
    {
        let parts: Vec<&str> = default_provider.split(':').collect();
        println!("Default provider: {default_provider}");
        return Some((parts[0].to_string(), parts.get(1).map(|s| s.to_string())));
    }

    if let Some(provider_string) = args.provider_or_key.clone() {
        let parts: Vec<&str> = provider_string.split(':').collect();
        return Some((parts[0].to_string(), parts.get(1).map(|s| s.to_string())));
    }

    args.provider
        .clone()
        .map(|provider| (provider, args.model.clone()))
}

/// Retrieves the appropriate API key for the specified backend
///
/// # Arguments
///
/// * `backend` - The LLM backend to get the API key for
/// * `args` - Command line arguments that may contain an API key
///
/// # Returns
///
/// * `Some(String)` - The API key if found
/// * `None` - If no API key could be found
fn get_api_key(backend: &LLMBackend, args: &CliArgs) -> Option<String> {
    args.api_key.clone().or_else(|| {
        let store = SecretStore::new().ok()?;
        let key = backend_env_key(backend)?;
        store.get(key).cloned().or_else(|| std::env::var(key).ok())
    })
}

fn backend_env_key(backend: &LLMBackend) -> Option<&'static str> {
    match backend {
        LLMBackend::OpenAI => Some("OPENAI_API_KEY"),
        LLMBackend::Anthropic => Some("ANTHROPIC_API_KEY"),
        LLMBackend::DeepSeek => Some("DEEPSEEK_API_KEY"),
        LLMBackend::XAI => Some("XAI_API_KEY"),
        LLMBackend::Google => Some("GOOGLE_API_KEY"),
        LLMBackend::Groq => Some("GROQ_API_KEY"),
        LLMBackend::AzureOpenAI => Some("AZURE_OPENAI_API_KEY"),
        LLMBackend::Cohere => Some("COHERE_API_KEY"),
        LLMBackend::Mistral => Some("MISTRAL_API_KEY"),
        LLMBackend::OpenRouter => Some("OPENROUTER_API_KEY"),
        LLMBackend::HuggingFace => Some("HF_TOKEN"),
        LLMBackend::Ollama
        | LLMBackend::Phind
        | LLMBackend::ElevenLabs
        | LLMBackend::AwsBedrock => None,
    }
}

/// Processes input data and creates appropriate chat messages
///
/// # Arguments
///
/// * `input` - Binary input data that might contain an image
/// * `prompt` - Text prompt to include in the message
///
/// # Returns
///
/// * `Vec<ChatMessage>` - Vector of chat messages ready to be sent to the LLM
fn process_input(input: &[u8], prompt: String) -> Vec<ChatMessage> {
    let mut messages = Vec::new();

    if let Some(mime) = detect_image_mime(input) {
        messages.push(ChatMessage::user().content(prompt).build());
        messages.push(ChatMessage::user().image(mime, input.to_vec()).build());
    } else if !input.is_empty() {
        let input_str = String::from_utf8_lossy(input);
        messages.push(
            ChatMessage::user()
                .content(format!("{prompt}\n\n{input_str}"))
                .build(),
        );
    } else {
        messages.push(ChatMessage::user().content(prompt).build());
    }

    messages
}

/// Main entry point for the LLM CLI application
///
/// Handles command parsing, provider configuration, and interactive chat functionality.
/// Supports various commands for managing secrets and default providers.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "logging")]
    {
        let _ = env_logger::try_init();
    }
    let args = CliArgs::parse();

    if let Some(cmd) = args.command.as_deref() {
        match cmd {
            "set" => {
                if let (Some(key), Some(value)) = (
                    args.provider_or_key.as_deref(),
                    args.prompt_or_value.as_deref(),
                ) {
                    let mut store = SecretStore::new()?;
                    store.set(key, value)?;
                    println!("{} Secret '{}' has been set.", "✓".bright_green(), key);
                    return Ok(());
                }
                eprintln!("{} Usage: llm set <key> <value>", "Error:".bright_red());
                return Ok(());
            }
            "get" => {
                if let Some(key) = args.provider_or_key.as_deref() {
                    let store = SecretStore::new()?;
                    match store.get(key) {
                        Some(value) => println!("{key}: {value}"),
                        None => println!("{} Secret '{}' not found", "!".bright_yellow(), key),
                    }
                    return Ok(());
                }
                eprintln!("{} Usage: llm get <key>", "Error:".bright_red());
                return Ok(());
            }
            "delete" => {
                if let Some(key) = args.provider_or_key.as_deref() {
                    let mut store = SecretStore::new()?;
                    store.delete(key)?;
                    println!("{} Secret '{}' has been deleted.", "✓".bright_green(), key);
                    return Ok(());
                }
                eprintln!("{} Usage: llm delete <key>", "Error:".bright_red());
                return Ok(());
            }
            "chat" => {}
            "default" => {
                if let Some(provider) = args.provider_or_key.as_deref() {
                    let mut store = SecretStore::new()?;
                    store.set_default_provider(provider)?;
                    return Ok(());
                } else if args.prompt_or_value.is_none() {
                    let store = SecretStore::new()?;
                    match store.get_default_provider() {
                        Some(provider) => println!("Default provider: {provider}"),
                        None => println!("{} No default provider set", "!".bright_yellow()),
                    }
                    return Ok(());
                }
                eprintln!(
                    "{} Usage: llm default <provider:model>",
                    "Error:".bright_red()
                );
                return Ok(());
            }
            _ => {}
        }
    }

    let (provider_name, model_name) = get_provider_info(&args)
        .ok_or("No provider specified. Use --provider, provider:model argument, or set a default provider with 'llm default <provider:model>'")?;

    let backend =
        LLMBackend::from_str(&provider_name).map_err(|e| format!("Invalid provider: {e}"))?;

    let mut builder = LLMBuilder::new().backend(backend.clone());

    if let Some(model) = model_name.or(args.model.clone()) {
        builder = builder.model(model);
    }

    if let Some(system) = args.system.clone() {
        builder = builder.system(system);
    }

    if let Some(key) = get_api_key(&backend, &args) {
        builder = builder.api_key(key);
    }

    if let Some(url) = args.base_url.clone() {
        builder = builder.base_url(url);
    }

    if let Some(temp) = args.temperature {
        builder = builder.temperature(temp);
    }

    if let Some(mt) = args.max_tokens {
        builder = builder.max_tokens(mt);
    }

    let provider = builder
        .build()
        .map_err(|e| format!("Failed to build provider: {e}"))?;

    let is_pipe = !io::stdin().is_terminal();

    if is_pipe || args.prompt_or_value.is_some() {
        let mut input = Vec::new();
        io::stdin().read_to_end(&mut input)?;

        let prompt = if let Some(p) = args.prompt_or_value {
            p
        } else {
            String::from_utf8_lossy(&input).to_string()
        };

        let messages = process_input(&input, prompt);

        match provider.chat(&messages).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    println!("{text}");
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
        }
        return Ok(());
    }

    println!("{}", "llm - Interactive Chat".bright_cyan());
    println!("Provider: {}", provider_name.bright_green());
    println!("{}", "Type 'exit' to quit".bright_black());
    println!("{}", "─".repeat(50).bright_black());

    let mut rl = DefaultEditor::new()?;
    let mut messages: Vec<ChatMessage> = Vec::new();

    loop {
        io::stdout().flush()?;
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.to_lowercase() == "exit" {
                    println!("{}", "👋 Goodbye!".bright_cyan());
                    break;
                }
                let _ = rl.add_history_entry(trimmed);

                let user_message = ChatMessage::user().content(trimmed.to_string()).build();
                messages.push(user_message);

                let mut sp =
                    Spinner::new(Spinners::Dots12, "Thinking...".bright_magenta().to_string());

                match provider.chat(&messages).await {
                    Ok(response) => {
                        sp.stop();
                        print!("\r\x1B[K");
                        if let Some(text) = response.text() {
                            println!("{} {}", "> Assistant:".bright_green(), text);
                            let assistant_message = ChatMessage::assistant().content(text).build();
                            messages.push(assistant_message);
                        } else {
                            println!("{}", "> Assistant: (no response)".bright_red());
                        }
                        println!("{}", "─".repeat(50).bright_black());
                    }
                    Err(e) => {
                        sp.stop();
                        eprintln!("{} {}", "Error:".bright_red(), e);
                        println!("{}", "─".repeat(50).bright_black());
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                println!("\n{}", "👋 Goodbye!".bright_cyan());
                break;
            }
            Err(err) => {
                eprintln!("{} {:?}", "Error:".bright_red(), err);
                break;
            }
        }
    }

    Ok(())
}

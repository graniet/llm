use clap::Parser;
use llm::builder::{LLMBackend, LLMBuilder};
use llm::chat::{ChatMessage, ImageMime};
use llm::secret_store::SecretStore;

#[path = "tests/mod.rs"]
mod tests;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::str::FromStr;
use std::io::{self, Write, Read, IsTerminal};
use colored::*;
use spinners::{Spinner, Spinners};
use comfy_table::{Table, ContentArrangement, Cell};
use comfy_table::presets::UTF8_FULL;

/// Command line arguments for the LLM CLI
#[derive(Parser)]
#[clap(name = "llm", about = "Interactive CLI interface for chatting with LLM providers", allow_hyphen_values = true)]
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
    if let Some(default_provider) = SecretStore::new().ok().and_then(|store| store.get_default_provider().cloned()) {
        let parts: Vec<&str> = default_provider.split(':').collect();
        // Only show default provider in interactive mode
    if io::stdin().is_terminal() && !matches!(args.command.as_deref(), Some("set") | Some("get") | Some("delete") | Some("default")) {
        println!("Default provider: {}", default_provider);
    }
        return Some((parts[0].to_string(), parts.get(1).map(|s| s.to_string())));
    }
    
    if let Some(provider_string) = args.provider_or_key.clone() {
        let parts: Vec<&str> = provider_string.split(':').collect();
        return Some((parts[0].to_string(), parts.get(1).map(|s| s.to_string())));
    }
    
    args.provider.clone().map(|provider| (provider, args.model.clone()))
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
        match backend {
            LLMBackend::OpenAI => store.get("OPENAI_API_KEY")
                .cloned()
                .or_else(|| std::env::var("OPENAI_API_KEY").ok()),
            LLMBackend::Anthropic => store.get("ANTHROPIC_API_KEY")
                .cloned()
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok()),
            LLMBackend::DeepSeek => store.get("DEEPSEEK_API_KEY")
                .cloned()
                .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok()),
            LLMBackend::XAI => store.get("XAI_API_KEY")
                .cloned()
                .or_else(|| std::env::var("XAI_API_KEY").ok()),
            LLMBackend::Google => store.get("GOOGLE_API_KEY")
                .cloned()
                .or_else(|| std::env::var("GOOGLE_API_KEY").ok()),
            LLMBackend::Groq => store.get("GROQ_API_KEY")
                .cloned()
                .or_else(|| std::env::var("GROQ_API_KEY").ok()),
            LLMBackend::Ollama => None,
            LLMBackend::Phind => None,
        }
    })
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
    
    if !input.is_empty() && detect_image_mime(input).is_some() {
        let mime = detect_image_mime(input).unwrap();
        messages.push(ChatMessage::user().content(prompt).build());
        messages.push(ChatMessage::user().image(mime, input.to_vec()).build());
    } else if !input.is_empty() {
        let input_str = String::from_utf8_lossy(input);
        messages.push(ChatMessage::user()
            .content(format!("{}\n\n{}", prompt, input_str))
            .build());
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
    let args = CliArgs::parse();

    if let Some(cmd) = args.command.as_deref() {
        match cmd {
            "set" => {
                if let (Some(key), Some(value)) = (args.provider_or_key.as_deref(), args.prompt_or_value.as_deref()) {
                    let mut store = SecretStore::new()?;
                    store.set(key, value)?;
                    // Create success table for secret setting
                    let mut success_table = Table::new();
                    success_table
                        .load_preset(UTF8_FULL)
                        .set_content_arrangement(ContentArrangement::Dynamic)
                        .set_width(80);
                    
                    success_table.add_row(vec![
                        Cell::new(format!("✅ Secret '{}' has been set.", key)).fg(comfy_table::Color::Green)
                    ]);
                    
                    println!("{}", success_table);
                    return Ok(());
                }
                // Create usage error table
                let mut error_table = Table::new();
                error_table
                    .load_preset(UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_width(80);
                
                error_table.add_row(vec![
                    Cell::new("❌ Error").fg(comfy_table::Color::Red).add_attribute(comfy_table::Attribute::Bold)
                ]);
                
                error_table.add_row(vec![
                    Cell::new("Usage: llm set <key> <value>").fg(comfy_table::Color::Red)
                ]);
                
                println!("{}", error_table);
                return Ok(());
            }
            "get" => {
                if let Some(key) = args.provider_or_key.as_deref() {
                    let store = SecretStore::new()?;
                    match store.get(key) {
                        Some(value) => println!("{}: {}", key, value),
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
                        Some(provider) => println!("Default provider: {}", provider),
                        None => println!("{} No default provider set", "!".bright_yellow()),
                    }
                    return Ok(());
                }
                eprintln!("{} Usage: llm default <provider:model>", "Error:".bright_red());
                return Ok(());
            }
            _ => {}
        }
    }

    let (provider_name, model_name) = get_provider_info(&args)
        .ok_or_else(|| "No provider specified. Use --provider, provider:model argument, or set a default provider with 'llm default <provider:model>'")?;

    let backend = LLMBackend::from_str(&provider_name)
        .map_err(|e| format!("Invalid provider: {}", e))?;

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

    let provider = builder.build()
        .map_err(|e| format!("Failed to build provider: {}", e))?;

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

        // In piped mode, don't show spinner
        match provider.chat(&messages).await {
            Ok(response) => {
                if let Some(text) = response.text() {
                    // For piped input/output, just print the raw text without table formatting
                    println!("{}", text);
                }
            }
            Err(e) => {
                // For piped input/output, use simple error format
                eprintln!("Error: {}", e);
            }
        }
        return Ok(());
    }

    // Create welcome header with comfy_table
    let mut header_table = Table::new();
    header_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(80);
    
    header_table.add_row(vec![
        Cell::new("llm - Interactive Chat").fg(comfy_table::Color::Cyan).add_attribute(comfy_table::Attribute::Bold)
    ]);
    
    header_table.add_row(vec![
        Cell::new(format!("Provider: {}", provider_name)).fg(comfy_table::Color::Green)
    ]);
    
    header_table.add_row(vec![
        Cell::new("Type 'exit' to quit").fg(comfy_table::Color::Grey)
    ]);
    
    println!("{}", header_table);

    let mut rl = DefaultEditor::new()?;
    let mut messages: Vec<ChatMessage> = Vec::new();

    loop {
        io::stdout().flush()?;
        let readline = rl.readline("💬 > ");
        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.to_lowercase() == "exit" {
                    // Create goodbye message table
    let mut goodbye_table = Table::new();
    goodbye_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(80);
    
    goodbye_table.add_row(vec![
        Cell::new("👋 Goodbye!").fg(comfy_table::Color::Cyan).add_attribute(comfy_table::Attribute::Bold)
    ]);
    
    println!("{}", goodbye_table);
                    break;
                }
                let _ = rl.add_history_entry(trimmed);

                let user_message = ChatMessage::user().content(trimmed.to_string()).build();
                messages.push(user_message);

                let mut sp = Spinner::new(Spinners::Dots12, "🤔 Thinking...".bright_magenta().to_string());

                match provider.chat(&messages).await {
                    Ok(response) => {
                        sp.stop();
                        print!("\r\x1B[K");
                        if let Some(text) = response.text() {
                            // Create response table
                            let mut response_table = Table::new();
                            response_table
                                .load_preset(UTF8_FULL)
                                .set_content_arrangement(ContentArrangement::Dynamic)
                                .set_width(80);
                            
                            // Add assistant header row
                            response_table.add_row(vec![
                                Cell::new("📢 Assistant").fg(comfy_table::Color::Green).add_attribute(comfy_table::Attribute::Bold)
                            ]);
                            
                            // Clone the text to avoid ownership issues
                            let text_clone = text.clone();
                            
                            // Add response content row
                            response_table.add_row(vec![
                                Cell::new(text_clone)
                            ]);
                            
                            println!("{}", response_table);
                            
                            let assistant_message =
                                ChatMessage::assistant().content(text).build();
                            messages.push(assistant_message);
                        } else {
                            // Create error table for no response
                            let mut error_table = Table::new();
                            error_table
                                .load_preset(UTF8_FULL)
                                .set_content_arrangement(ContentArrangement::Dynamic)
                                .set_width(80);
                            
                            error_table.add_row(vec![
                                Cell::new("❌ Assistant: (no response)").fg(comfy_table::Color::Red)
                            ]);
                            
                            println!("{}", error_table);
                        }
                    }
                    Err(e) => {
                        sp.stop();
                        
                        // Create error table
                        let mut error_table = Table::new();
                        error_table
                            .load_preset(UTF8_FULL)
                            .set_content_arrangement(ContentArrangement::Dynamic)
                            .set_width(80);
                        
                        error_table.add_row(vec![
                            Cell::new("❌ Error").fg(comfy_table::Color::Red).add_attribute(comfy_table::Attribute::Bold)
                        ]);
                        
                        error_table.add_row(vec![
                            Cell::new(e.to_string()).fg(comfy_table::Color::Red)
                        ]);
                        
                        println!("{}", error_table);
                    }
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => {
                // Create goodbye message table
                let mut goodbye_table = Table::new();
                goodbye_table
                    .load_preset(UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_width(80);
                
                goodbye_table.add_row(vec![
                    Cell::new("👋 Goodbye!").fg(comfy_table::Color::Cyan).add_attribute(comfy_table::Attribute::Bold)
                ]);
                
                println!("\n{}", goodbye_table);
                break;
            }
            Err(err) => {
                // Create error table
                let mut error_table = Table::new();
                error_table
                    .load_preset(UTF8_FULL)
                    .set_content_arrangement(ContentArrangement::Dynamic)
                    .set_width(80);
                
                error_table.add_row(vec![
                    Cell::new("❌ Error").fg(comfy_table::Color::Red).add_attribute(comfy_table::Attribute::Bold)
                ]);
                
                error_table.add_row(vec![
                    Cell::new(format!("{:?}", err)).fg(comfy_table::Color::Red)
                ]);
                
                println!("{}", error_table);
                break;
            }
        }
    }

    Ok(())
}
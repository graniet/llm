use clap::{Parser, Subcommand};
use llm::builder::{LLMBackend, LLMBuilder};
use llm::chain::{
    ChainStepMode, LLMRegistryBuilder, MultiChainStepBuilder, MultiChainStepMode,
    MultiPromptChain,
};

#[path = "tests/mod.rs"]
mod tests;
use llm::ToolCall;
use llm::secret_store::SecretStore;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use comfy_table::{Table, ContentArrangement, Cell};
use comfy_table::presets::UTF8_FULL;
use colored::*;
use spinners::{Spinner, Spinners};
use serde::{Deserialize, Serialize};
use std::io::{self, IsTerminal, Read, Write};
use rustyline::DefaultEditor;
use chrono::{DateTime, Local};
use textwrap;

/// Command line arguments for the LLM Chain CLI
#[derive(Parser)]
#[clap(name = "llm-chain", about = "CLI for running LLM chains with multiple steps", allow_hyphen_values = true)]
struct CliArgs {
    /// Subcommand to execute
    #[command(subcommand)]
    command: Option<Commands>,

    /// Provider string in format "provider:model"
    #[arg(long)]
    provider: Option<String>,

    /// Path to a YAML or JSON chain definition file
    #[arg(long)]
    file: Option<PathBuf>,

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
    
    /// Output results in JSON format
    #[arg(long)]
    json: bool,
    
    /// Run in interactive mode
    #[arg(long)]
    interactive: bool,
    
    /// Specific steps to make interactive (comma-separated)
    #[arg(long)]
    interactive_steps: Option<String>,
    
    /// Save interaction history to file
    #[arg(long)]
    save_history: Option<PathBuf>,
    
    /// Replay interaction from saved history file
    #[arg(long)]
    replay: Option<PathBuf>,
}

/// Subcommands for the LLM Chain CLI
#[derive(Subcommand)]
enum Commands {
    /// Run a chain from a file or interactive input
    Run {
        /// Path to a YAML or JSON chain definition file
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Create a new chain definition file
    Create {
        /// Path to save the new chain definition file
        #[arg(long)]
        output: PathBuf,
    },
    /// View providers available for use in chains
    Providers,
}

/// Step configuration for a chain
#[derive(Debug, Serialize, Deserialize, Clone)]
struct StepConfig {
    /// Step ID
    id: String,
    /// Prompt template with {{variable}} placeholders
    template: String,
    /// Provider ID (for multi-provider chains)
    provider_id: Option<String>,
    /// Execution mode (chat or completion)
    #[serde(default = "default_mode")]
    mode: String,
    /// Temperature parameter (0.0-1.0)
    temperature: Option<f32>,
    /// Maximum tokens to generate
    max_tokens: Option<u32>,
    /// Condition that determines whether to run this step
    condition: Option<String>,
    /// Whether this step should pause for user interaction in interactive mode
    #[serde(default = "default_interactive")]
    interactive: bool,
}

/// Returns the default mode for step configuration
fn default_mode() -> String {
    "chat".to_string()
}

/// Returns the default interactive setting for step configuration
fn default_interactive() -> bool {
    false
}

/// Interactive settings for a chain 
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct InteractiveConfig {
    /// Whether to automatically start in interactive mode
    #[serde(default = "default_false")]
    auto_start: bool,
    /// Default steps to make interactive
    #[serde(default)]
    default_steps: Vec<String>,
    /// Timeout in seconds before automatically continuing
    #[serde(default = "default_timeout")]
    timeout: u32,
    /// Path to save interaction history
    #[serde(default)]
    save_path: Option<PathBuf>,
}

/// Represents a user interaction choice during interactive mode
#[derive(Debug, Clone, Serialize, Deserialize)]
enum InteractiveChoice {
    /// Accept the result and continue to the next step
    Accept,
    /// Edit the current response
    EditResponse(String),
    /// Modify the next prompt
    ModifyPrompt(String),
    /// Skip to a specific step
    SkipToStep(String),
    /// View current variables
    ViewVars,
    /// Quit interactive mode
    Quit,
}

/// History of interactions for saving/replaying
#[derive(Debug, Serialize, Deserialize)]
struct InteractionHistory {
    /// Chain configuration used
    chain_config: ChainConfig,
    /// Initial input
    input: Option<String>,
    /// History of all interactions
    interactions: Vec<StepInteraction>,
    /// Timestamp when this history was created (stored as ISO 8601 string)
    #[serde(default = "default_timestamp")]
    timestamp: String,
}

/// Returns the current timestamp in ISO 8601 format
fn default_timestamp() -> String {
    Local::now().to_rfc3339()
}

/// Record of a single step interaction
#[derive(Debug, Serialize, Deserialize)]
struct StepInteraction {
    /// Step ID
    step_id: String,
    /// LLM response
    response: String,
    /// User's choice
    choice: InteractiveChoice,
}

/// Returns false as a default value
fn default_false() -> bool {
    false
}

/// Returns the default timeout value in seconds
fn default_timeout() -> u32 {
    300 // 5 minutes
}

/// Chain configuration
#[derive(Debug, Serialize, Deserialize, Clone)]
struct ChainConfig {
    /// Chain name
    name: String,
    /// Chain description
    description: Option<String>,
    /// Default provider to use
    default_provider: Option<String>,
    /// Steps in the chain
    steps: Vec<StepConfig>,
    /// Input variable name for piped input (default: "input")
    #[serde(default = "default_input_var")]
    input_var: String,
    /// Interactive mode configuration
    #[serde(default)]
    interactive_config: InteractiveConfig,
}

/// Returns the default input variable name
fn default_input_var() -> String {
    "input".to_string()
}
/// Main entry point for the LLM Chain CLI application
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    match &args.command {
        Some(Commands::Providers) => {
            display_providers();
            return Ok(());
        }
        Some(Commands::Create { output }) => {
            create_chain_template(output)?;
            return Ok(());
        }
        _ => {
            // Continue with chain execution
        }
    }
    
    // Check if there's input from a pipe
    let is_pipe = !io::stdin().is_terminal();
    let piped_input = if is_pipe {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)?;
        Some(buffer)
    } else {
        None
    };

    // Load chain configuration from file or use default
    let mut chain_config = if let Some(ref file_path) = args.file {
        load_chain_config(file_path)?
    } else if let Some(Commands::Run { file }) = &args.command {
        if let Some(ref file_path) = file {
            load_chain_config(file_path)?
        } else {
            return Err("No chain configuration provided. Use --file or run llm-chain create to make one.".into());
        }
    } else {
        return Err("No chain configuration provided. Use --file or run llm-chain create to make one.".into());
    };
    
    // Apply interactive mode settings from command line args if provided
    if args.interactive {
        // If --interactive is specified, override the auto_start setting
        chain_config.interactive_config.auto_start = true;
        
        // If --interactive-steps is specified, use those steps instead of the default ones
        if let Some(steps_str) = &args.interactive_steps {
            let steps: Vec<String> = steps_str.split(',')
                .map(|s| s.trim().to_string())
                .collect();
            chain_config.interactive_config.default_steps = steps;
        }
        
        // Apply interactive flag to all steps mentioned in default_steps
        for step in &mut chain_config.steps {
            if chain_config.interactive_config.default_steps.contains(&step.id) {
                step.interactive = true;
            }
        }
    }
    
    // Set save history path from command line if provided
    if let Some(save_path) = &args.save_history {
        chain_config.interactive_config.save_path = Some(save_path.clone());
    }
    
    // Load and replay history if provided
    if let Some(replay_path) = &args.replay {
        // Load the saved interaction history
        let history_content = fs::read_to_string(replay_path)?;
        let history: InteractionHistory = serde_json::from_str(&history_content)?;
        
        println!("🎬 Replaying interaction history from {}", replay_path.display());
        
        // Apply the loaded chain configuration
        chain_config = history.chain_config;
        
        // TODO: Implement history replay functionality
        // This would require simulating the interaction choices
        println!("Note: Replay functionality is still under development");
    }

    // Get provider info
    let (provider_name, model_name) = get_provider_info(&args, &chain_config)?;
    let backend = LLMBackend::from_str(&provider_name)
        .map_err(|e| format!("Invalid provider: {}", e))?;

    // Build provider
    let mut builder = LLMBuilder::new().backend(backend.clone());

    if let Some(model) = model_name {
        builder = builder.model(model);
    }

    if let Some(key) = get_api_key(&backend, &args) {
        builder = builder.api_key(key);
    }

    if let Some(url) = args.base_url {
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

    // If running a multi-provider chain, establish the registry
    if has_multiple_providers(&chain_config) {
        run_multi_provider_chain(&chain_config, provider, provider_name, piped_input, args.json).await?;
    } else {
        run_single_provider_chain(&chain_config, provider, piped_input, args.json).await?;
    }

    Ok(())
}

/// Checks if the chain configuration uses multiple providers
fn has_multiple_providers(config: &ChainConfig) -> bool {
    config.steps.iter().any(|step| step.provider_id.is_some())
}

/// Populates system variables with current environment information
fn populate_system_variables() -> HashMap<String, String> {
    let mut vars = HashMap::new();
    
    // Current date and time
    let now: DateTime<Local> = Local::now();
    vars.insert("sys.date".to_string(), now.format("%Y-%m-%d").to_string());
    vars.insert("sys.time".to_string(), now.format("%H:%M:%S").to_string());
    vars.insert("sys.datetime".to_string(), now.format("%Y-%m-%d %H:%M:%S").to_string());
    vars.insert("sys.timestamp".to_string(), now.timestamp().to_string());
    
    // OS information
    vars.insert("sys.os".to_string(), std::env::consts::OS.to_string());
    vars.insert("sys.arch".to_string(), std::env::consts::ARCH.to_string());
    
    // User information if available
    if let Ok(user) = std::env::var("USER") {
        vars.insert("sys.user".to_string(), user);
    } else if let Ok(username) = std::env::var("USERNAME") {
        vars.insert("sys.user".to_string(), username);
    }
    
    // Hostname if available
    if let Ok(hostname) = hostname::get() {
        if let Ok(hostname_str) = hostname.into_string() {
            vars.insert("sys.hostname".to_string(), hostname_str);
        }
    }
    
    vars
}

/// Handles interactive prompt editing and choice selection
fn handle_interactive_prompt(
    step_id: &str,
    response: &str,
    memory: &HashMap<String, String>,
    config: &ChainConfig,
) -> Result<InteractiveChoice, Box<dyn std::error::Error>> {
    // Create a visual separator
    let separator = "─".repeat(100);
    println!("\n{}", separator.bright_blue());
    
    // Show step header with progress indicator
    let current_step_idx = config.steps.iter().position(|s| s.id == step_id).unwrap_or(0);
    let step_progress = format!("Step {}/{}", current_step_idx + 1, config.steps.len());
    
    println!("🔗 {}: {} - {}", "INTERACTIVE MODE".bright_magenta().bold(), 
        step_id.bright_cyan().bold(), 
        step_progress.yellow());
    
    // Display response in a nicely formatted box
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(100);
    
    // Add a header
    table.add_row(vec![
        Cell::new("🤖 LLM RESPONSE").fg(comfy_table::Color::Green).add_attribute(comfy_table::Attribute::Bold)
    ]);
    
    // Add the response content with word wrapping
    let wrapped_response = textwrap::fill(response, 95);
    table.add_row(vec![
        Cell::new(wrapped_response).fg(comfy_table::Color::White)
    ]);
    
    println!("{}", table);
    
    // Show keyboard shortcuts in a more appealing way
    let mut menu_table = Table::new();
    menu_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(100);
    
    menu_table.set_header(vec![
        Cell::new("KEY").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("ACTION").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("DESCRIPTION").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold)
    ]);
    
    menu_table.add_row(vec![
        Cell::new("A or 1").fg(comfy_table::Color::Green),
        Cell::new("Accept").fg(comfy_table::Color::White).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("Continue to next step"),
    ]);
    
    menu_table.add_row(vec![
        Cell::new("E or 2").fg(comfy_table::Color::Green),
        Cell::new("Edit").fg(comfy_table::Color::White).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("Modify the current response"),
    ]);
    
    menu_table.add_row(vec![
        Cell::new("M or 3").fg(comfy_table::Color::Green),
        Cell::new("Modify").fg(comfy_table::Color::White).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("Change the next prompt"),
    ]);
    
    menu_table.add_row(vec![
        Cell::new("S or 4").fg(comfy_table::Color::Green),
        Cell::new("Skip").fg(comfy_table::Color::White).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("Jump to different step"),
    ]);
    
    menu_table.add_row(vec![
        Cell::new("V or 5").fg(comfy_table::Color::Green),
        Cell::new("Variables").fg(comfy_table::Color::White).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("View current variable values"),
    ]);
    
    menu_table.add_row(vec![
        Cell::new("H or 6").fg(comfy_table::Color::Green),
        Cell::new("Help").fg(comfy_table::Color::White).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("Show help information"),
    ]);
    
    menu_table.add_row(vec![
        Cell::new("Q or 7").fg(comfy_table::Color::Green),
        Cell::new("Quit").fg(comfy_table::Color::White).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("Exit the chain execution"),
    ]);
    
    println!("{}", menu_table);
    
    // Display history saving information if enabled
    if config.interactive_config.save_path.is_some() {
        println!("💾 {}", "Interaction history will be saved when chain completes".bright_blue());
    }
    
    // Use readline for better input handling
    let mut rl = DefaultEditor::new()?;
    let prompt = format!("{} ", "Your choice:".bright_yellow().bold());
    
    loop {
        print!("\n{}", prompt);
        io::stdout().flush()?;
        
        let readline = rl.readline("");
        match readline {
            Ok(line) => {
                let choice = line.trim().to_lowercase();
                
                match choice.as_str() {
                    "a" | "1" => {
                        println!("✅ {}", "Continuing with current response...".green());
                        return Ok(InteractiveChoice::Accept);
                    },
                    "e" | "2" => {
                        println!("✏️ {}", "Opening editor to modify response...".yellow());
                        // Open editor to edit response
                        let edited = edit::edit(response)?;
                        return Ok(InteractiveChoice::EditResponse(edited));
                    },
                    "m" | "3" => {
                        // Find the next step
                        let current_index = config.steps.iter().position(|s| s.id == step_id).unwrap_or(0);
                        if current_index + 1 < config.steps.len() {
                            let next_step = &config.steps[current_index + 1];
                            let next_template = apply_template(&next_step.template, memory);
                            
                            println!("✏️ {} {}", "Opening editor to modify prompt for next step:".yellow(), 
                                     next_step.id.bright_cyan());
                                     
                            // Open editor to modify next prompt
                            let edited = edit::edit(&next_template)?;
                            return Ok(InteractiveChoice::ModifyPrompt(edited));
                        } else {
                            println!("⚠️ {}", "This is the last step, no next prompt to modify.".bright_red());
                        }
                    },
                    "s" | "4" => {
                        // Show available steps in a table
                        let mut steps_table = Table::new();
                        steps_table
                            .load_preset(UTF8_FULL)
                            .set_content_arrangement(ContentArrangement::Dynamic)
                            .set_width(80);
                            
                        steps_table.set_header(vec![
                            Cell::new("#").fg(comfy_table::Color::Yellow),
                            Cell::new("STEP ID").fg(comfy_table::Color::Yellow),
                            Cell::new("STATUS").fg(comfy_table::Color::Yellow)
                        ]);
                        
                        for (i, step) in config.steps.iter().enumerate() {
                            let status = if step.id == step_id { 
                                "CURRENT".bright_green() 
                            } else if i < current_step_idx {
                                "COMPLETED".bright_blue()
                            } else {
                                "PENDING".normal()
                            };
                            
                            steps_table.add_row(vec![
                                Cell::new(format!("{}", i+1)).fg(comfy_table::Color::White),
                                Cell::new(&step.id).fg(comfy_table::Color::Cyan),
                                Cell::new(status.to_string())
                            ]);
                        }
                        
                        println!("\n{}", steps_table);
                        
                        print!("{} ", "Jump to step #:".bright_yellow());
                        io::stdout().flush()?;
                        
                        let step_choice = rl.readline("");
                        if let Ok(step_input) = step_choice {
                            if let Ok(idx) = step_input.trim().parse::<usize>() {
                                if idx > 0 && idx <= config.steps.len() {
                                    let target_step = &config.steps[idx-1];
                                    println!("🔄 {} {}", "Jumping to step:".yellow(), target_step.id.bright_cyan());
                                    return Ok(InteractiveChoice::SkipToStep(target_step.id.clone()));
                                }
                            }
                            println!("❌ {}", "Invalid step number.".bright_red());
                        }
                    },
                    "v" | "5" => {
                        // Display variables in a table
                        let mut vars_table = Table::new();
                        vars_table
                            .load_preset(UTF8_FULL)
                            .set_content_arrangement(ContentArrangement::Dynamic)
                            .set_width(100);
                            
                        vars_table.set_header(vec![
                            Cell::new("VARIABLE").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold),
                            Cell::new("TYPE").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold),
                            Cell::new("VALUE").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold)
                        ]);
                        
                        let mut vars: Vec<(&String, &String)> = memory.iter().collect();
                        vars.sort_by(|a, b| a.0.cmp(b.0));
                        let vars_count = vars.len();
                        
                        for (k, v) in &vars {
                            let var_type = if k.starts_with("sys.") { 
                                "SYSTEM".bright_blue() 
                            } else if *k == &config.input_var {
                                "INPUT".bright_magenta()
                            } else if config.steps.iter().any(|s| &s.id == *k) {
                                "STEP".bright_green()
                            } else {
                                "CUSTOM".bright_yellow()
                            };
                            
                            // Limit display length for very long values
                            let value_display = if v.len() > 100 {
                                format!("{}...", &v[0..97])
                            } else {
                                v.to_string()
                            };
                            
                            vars_table.add_row(vec![
                                Cell::new(*k).fg(comfy_table::Color::Cyan),
                                Cell::new(var_type.to_string()),
                                Cell::new(&value_display)
                            ]);
                        }
                        
                        println!("\n{}", vars_table);
                        println!("📊 {} {}", "Total variables:".bright_yellow(), vars_count.to_string().bright_white());
                    },
                    "h" | "6" => {
                        println!("\n{}:", "Interactive Mode Help".bright_cyan().bold());
                        println!("- {}: {}", "Accept [A]".bright_green(), "Continue to the next step with the current response");
                        println!("- {}: {}", "Edit [E]".bright_green(), "Edit the current LLM response before continuing");
                        println!("- {}: {}", "Modify [M]".bright_green(), "Change the prompt for the next step");
                        println!("- {}: {}", "Skip [S]".bright_green(), "Jump to a specific step in the chain");
                        println!("- {}: {}", "Variables [V]".bright_green(), "See all current variable values");
                        println!("- {}: {}", "Help [H]".bright_green(), "Show this help message");
                        println!("- {}: {}", "Quit [Q]".bright_green(), "Exit the chain and return current results");
                        
                        // Show additional information about modes
                        println!("\n{}:", "Chain Information".bright_cyan().bold());
                        println!("- {}: {}", "Name".bright_yellow(), config.name);
                        if let Some(desc) = &config.description {
                            println!("- {}: {}", "Description".bright_yellow(), desc);
                        }
                        println!("- {}: {}", "Interactive Steps".bright_yellow(), 
                             config.interactive_config.default_steps.join(", "));
                    },
                    "q" | "7" => {
                        println!("👋 {}", "Exiting chain execution...".yellow());
                        return Ok(InteractiveChoice::Quit);
                    },
                    _ => println!("❌ {} Type 'h' for help.", "Invalid choice.".bright_red()),
                }
            },
            Err(_) => println!("❌ {}", "Error reading input. Please try again.".bright_red()),
        }
    }
}

/// Applies template with variable substitution
fn apply_template(template: &str, memory: &HashMap<String, String>) -> String {
    let mut result = template.to_string();
    for (k, v) in memory {
        let pattern = format!("{{{{{}}}}}", k);
        result = result.replace(&pattern, v);
    }
    result
}

/// Evaluates conditions for conditional steps
fn evaluate_condition(condition: &Option<String>, memory: &HashMap<String, String>) -> bool {
    // If no condition is specified, always run the step
    if let Some(condition) = condition {
        if condition.is_empty() {
            return true;
        }
        
        // Check if we're doing an equality comparison
        if let Some(equals_pos) = condition.find('=') {
            let var_name = condition[..equals_pos].trim().to_string();
            let expected_value = &condition[equals_pos+1..].trim();
            
            if let Some(actual_value) = memory.get(&var_name) {
                return actual_value == expected_value;
            }
            return false;
        }
        
        // Check if we're doing a contains check
        if condition.contains("contains") {
            let parts: Vec<&str> = condition.split("contains").collect();
            if parts.len() == 2 {
                let var_name = parts[0].trim().to_string();
                let search_value = parts[1].trim().trim_matches('"').trim_matches('\'');
                
                if let Some(actual_value) = memory.get(&var_name) {
                    return actual_value.contains(search_value);
                }
            }
            return false;
        }
        
        // Check for existence (non-empty)
        if condition.starts_with('!') {
            let var_name = condition[1..].trim().to_string();
            return !memory.contains_key(&var_name) || memory.get(&var_name).map_or(true, |v| v.is_empty());
        } else {
            let var_name = condition.trim().to_string();
            return memory.contains_key(&var_name) && memory.get(&var_name).map_or(false, |v| !v.is_empty());
        }
    } else {
        // No condition means always run
        return true;
    }
}

/// JSON response format for chain results
#[derive(Serialize)]
struct JsonResponse {
    /// Chain name
    chain_name: String,
    /// Individual step results
    steps: HashMap<String, String>,
    /// Final result (from the last step)
    result: String,
}

/// Runs a chain with a single provider
async fn run_single_provider_chain(
    config: &ChainConfig,
    provider: Box<dyn llm::LLMProvider>,
    piped_input: Option<String>,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_pipe = !io::stdout().is_terminal();
    
    if !is_pipe && !json_output {
        // Create welcome header
        let mut header_table = Table::new();
        header_table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(80);
        
        header_table.add_row(vec![
            Cell::new(format!("🔗 Running chain: {}", config.name)).fg(comfy_table::Color::Cyan).add_attribute(comfy_table::Attribute::Bold)
        ]);
        
        if let Some(desc) = &config.description {
            header_table.add_row(vec![
                Cell::new(desc)
            ]);
        }
        
        println!("{}", header_table);
    }

    let mut initial_memory = HashMap::new();
    let input_value = piped_input.as_ref().map_or_else(|| "".to_string(), |s| s.clone());
    initial_memory.insert(config.input_var.clone(), input_value);
    
    let system_vars = populate_system_variables();
    initial_memory.extend(system_vars);
    
    let mut memory_for_conditions = initial_memory.clone();
    
    let is_interactive = (!is_pipe && !json_output) || config.interactive_config.auto_start;
    let interactive_steps = config.interactive_config.default_steps.clone();
    
    let _interaction_history = if is_interactive {
        Some(InteractionHistory {
            chain_config: config.clone(),
            input: piped_input.clone(),
            interactions: Vec::new(),
            timestamp: Local::now().to_rfc3339(),
        })
    } else {
        None
    };
    
    let mut results = HashMap::new();
    
    let mut current_step_idx = 0;
    while current_step_idx < config.steps.len() {
        let step = &config.steps[current_step_idx];
        
        let should_run = evaluate_condition(&step.condition, &memory_for_conditions);
        if !should_run {
            if !is_pipe && !json_output {
                println!("⏭️  Skipping step '{}' (condition not met)", step.id);
            }
            current_step_idx += 1;
            continue;
        }
        
        let is_step_interactive = is_interactive && 
            (step.interactive || interactive_steps.contains(&step.id));
        
        let applied_template = apply_template(&step.template, &memory_for_conditions);
        
        let sp = if is_pipe || json_output || is_step_interactive {
            None
        } else {
            Some(Spinner::new(Spinners::Dots12, "🔄 Running chain...".bright_magenta().to_string()))
        };
        
        let mode = match step.mode.to_lowercase().as_str() {
            "completion" => ChainStepMode::Completion,
            _ => ChainStepMode::Chat,
        };
        
        let messages = vec![
            llm::chat::ChatMessage::user().content(applied_template.clone()).build()
        ];
        
        let mut temperature = None;
        if let Some(temp) = step.temperature {
            temperature = Some(temp);
        }
        
        let mut max_tokens = None;
        if let Some(mt) = step.max_tokens {
            max_tokens = Some(mt);
        }
        
        let response = match mode {
            ChainStepMode::Chat => {
                provider.chat(&messages).await
                    .map_err(|e| format!("Chain step '{}' failed: {}", step.id, e))?
            },
            ChainStepMode::Completion => {
                let mut req = llm::completion::CompletionRequest::new(applied_template);
                req.temperature = temperature;
                req.max_tokens = max_tokens;
                let response = provider.as_ref().complete(&req).await
                    .map_err(|e| format!("Chain step '{}' failed: {}", step.id, e))?;
                
                struct CompletionChatResponse {
                    text: String,
                }
                
                impl std::fmt::Debug for CompletionChatResponse {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(f, "CompletionChatResponse {{ text: {} }}", self.text)
                    }
                }
                
                impl std::fmt::Display for CompletionChatResponse {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(f, "{}", self.text)
                    }
                }
                
                impl llm::chat::ChatResponse for CompletionChatResponse {
                    fn text(&self) -> Option<String> {
                        Some(self.text.clone())
                    }
                    
                    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
                        None
                    }
                }
                
                Box::new(CompletionChatResponse { text: response.text })
            }
        };
        
        let mut response_text = response.text().unwrap_or_default().to_string();
        
        if let Some(mut spinner) = sp {
            spinner.stop();
            print!("\r\x1B[K");
        }
        
        if is_step_interactive {
            let user_choice = handle_interactive_prompt(&step.id, &response_text, &memory_for_conditions, config)?;
            
            match user_choice {
                InteractiveChoice::Accept => {
                    // Continue with the response as-is
                },
                InteractiveChoice::EditResponse(edited_text) => {
                    response_text = edited_text;
                },
                InteractiveChoice::ModifyPrompt(modified_prompt) => {
                    let updated_messages = vec![
                        llm::chat::ChatMessage::user().content(modified_prompt).build()
                    ];
                    
                    let new_response = provider.chat(&updated_messages).await
                        .map_err(|e| format!("Chain step '{}' (modified) failed: {}", step.id, e))?;
                    
                    response_text = new_response.text().unwrap_or_default().to_string();
                },
                InteractiveChoice::SkipToStep(target_step) => {
                    if let Some(idx) = config.steps.iter().position(|s| s.id == target_step) {
                        current_step_idx = idx;
                        continue;
                    }
                },
                InteractiveChoice::ViewVars => {},
                InteractiveChoice::Quit => {
                    return if json_output {
                        let json_response = JsonResponse {
                            chain_name: config.name.clone(),
                            steps: results.clone(),
                            result: results.values().last().cloned().unwrap_or_default(),
                        };
                        println!("{}", serde_json::to_string_pretty(&json_response)?);
                        Ok(())
                    } else {
                        display_chain_results(&results, &config.steps);
                        Ok(())
                    };
                }
            }
        }
        
        results.insert(step.id.clone(), response_text.clone());
        
        memory_for_conditions.insert(step.id.clone(), response_text);
        
        current_step_idx += 1;
    }
    
    if json_output {
        let final_result = if let Some(final_step) = config.steps.last() {
            results.get(&final_step.id).cloned().unwrap_or_default()
        } else {
            String::new()
        };
        
        let json_response = JsonResponse {
            chain_name: config.name.clone(),
            steps: results.clone(),
            result: final_result,
        };
        
        println!("{}", serde_json::to_string_pretty(&json_response)?);
    } else if is_pipe {
        if let Some(final_step) = config.steps.last() {
            if let Some(result) = results.get(&final_step.id) {
                println!("{}", result);
            }
        }
    } else {
        display_chain_results(&results, &config.steps);
    }
    
    Ok(())
}

/// Run a chain with multiple providers
async fn run_multi_provider_chain(
    config: &ChainConfig,
    default_provider: Box<dyn llm::LLMProvider>,
    default_provider_name: String,
    piped_input: Option<String>,
    json_output: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let is_pipe = !io::stdout().is_terminal();
    
    if !is_pipe && !json_output {
        let mut header_table = Table::new();
        header_table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_width(80);
        
        header_table.add_row(vec![
            Cell::new(format!("🔗 Running multi-provider chain: {}", config.name)).fg(comfy_table::Color::Cyan).add_attribute(comfy_table::Attribute::Bold)
        ]);
        
        if let Some(desc) = &config.description {
            header_table.add_row(vec![
                Cell::new(desc)
            ]);
        }
        
        println!("{}", header_table);
    }

    // Build registry with the default provider
    let registry_builder = LLMRegistryBuilder::new()
        .register(&default_provider_name, default_provider);
    
    let registry = registry_builder.build();
    let chain = MultiPromptChain::new(&registry);
    let mut initial_memory = HashMap::new();
    let input_value = piped_input.as_ref().map_or_else(|| "".to_string(), |s| s.clone());
    initial_memory.insert(config.input_var.clone(), input_value);
    
    let system_vars = populate_system_variables();
    initial_memory.extend(system_vars);
    
    let mut memory_for_conditions = initial_memory.clone();
    
    let _chain = chain.with_memory(initial_memory);
    
    let is_interactive = (!is_pipe && !json_output) || config.interactive_config.auto_start;
    let interactive_steps = config.interactive_config.default_steps.clone();
    
    let _interaction_history = if is_interactive {
        Some(InteractionHistory {
            chain_config: config.clone(),
            input: piped_input.clone(),
            interactions: Vec::new(),
            timestamp: Local::now().to_rfc3339(),
        })
    } else {
        None
    };
    
    let _current_chain = MultiPromptChain::new(&registry).with_memory(memory_for_conditions.clone());
    
    let mut current_step_idx = 0;
    let mut results = HashMap::new();
    
    while current_step_idx < config.steps.len() {
        let step = &config.steps[current_step_idx];
        
        let should_run = evaluate_condition(&step.condition, &memory_for_conditions);
        if !should_run {
            if !is_pipe && !json_output {
                println!("⏭️  Skipping step '{}' (condition not met)", step.id);
            }
            current_step_idx += 1;
            continue;
        }
        
        let is_step_interactive = is_interactive && 
            (step.interactive || interactive_steps.contains(&step.id));
        
        let applied_template = apply_template(&step.template, &memory_for_conditions);
        
        let sp = if is_pipe || json_output || is_step_interactive {
            None
        } else {
            Some(Spinner::new(Spinners::Dots12, "🔄 Running chain...".bright_magenta().to_string()))
        };
        
        let mode = match step.mode.to_lowercase().as_str() {
            "completion" => MultiChainStepMode::Completion,
            _ => MultiChainStepMode::Chat,
        };
        
        let provider_id = step.provider_id.as_deref().unwrap_or(&default_provider_name);
        
        let mut step_builder = MultiChainStepBuilder::new(mode.clone())
            .provider_id(provider_id)
            .id(&step.id)
            .template(&applied_template);
        
        if let Some(temp) = step.temperature {
            step_builder = step_builder.temperature(temp);
        }
        
        if let Some(mt) = step.max_tokens {
            step_builder = step_builder.max_tokens(mt);
        }
        
        let built_step = step_builder.build().map_err(|e| format!("Failed to build step: {}", e))?;
        
        let single_step_chain = MultiPromptChain::new(&registry)
            .with_memory(memory_for_conditions.clone())
            .step(built_step);
        
        let step_result = single_step_chain.run().await
            .map_err(|e| format!("Chain step '{}' failed: {}", step.id, e))?;
        
        let mut response_text = step_result.get(&step.id).cloned().unwrap_or_default();
        
        if let Some(mut spinner) = sp {
            spinner.stop();
            print!("\r\x1B[K");
        }
        
        if is_step_interactive {
            let user_choice = handle_interactive_prompt(&step.id, &response_text, &memory_for_conditions, config)?;
            
            match user_choice {
                InteractiveChoice::Accept => {},
                InteractiveChoice::EditResponse(edited_text) => {
                    response_text = edited_text;
                },
                InteractiveChoice::ModifyPrompt(modified_prompt) => {
                    let mut modified_step_builder = MultiChainStepBuilder::new(mode.clone())
                        .provider_id(provider_id)
                        .id(&step.id)
                        .template(&modified_prompt);
                    
                    if let Some(temp) = step.temperature {
                        modified_step_builder = modified_step_builder.temperature(temp);
                    }
                    
                    if let Some(mt) = step.max_tokens {
                        modified_step_builder = modified_step_builder.max_tokens(mt);
                    }
                    
                    let modified_built_step = modified_step_builder.build()
                        .map_err(|e| format!("Failed to build modified step: {}", e))?;
                    
                    let modified_single_step_chain = MultiPromptChain::new(&registry)
                        .with_memory(memory_for_conditions.clone())
                        .step(modified_built_step);
                    
                    let modified_step_result = modified_single_step_chain.run().await
                        .map_err(|e| format!("Chain step '{}' (modified) failed: {}", step.id, e))?;
                    
                    response_text = modified_step_result.get(&step.id).cloned().unwrap_or_default();
                },
                InteractiveChoice::SkipToStep(target_step) => {
                    if let Some(idx) = config.steps.iter().position(|s| s.id == target_step) {
                        current_step_idx = idx;
                        continue;
                    }
                },
                InteractiveChoice::ViewVars => {},
                InteractiveChoice::Quit => {                    
                    return if json_output {
                        let json_response = JsonResponse {
                            chain_name: config.name.clone(),
                            steps: results.clone(),
                            result: results.values().last().cloned().unwrap_or_default(),
                        };
                        println!("{}", serde_json::to_string_pretty(&json_response)?);
                        Ok(())
                    } else {
                        display_chain_results(&results, &config.steps);
                        Ok(())
                    };
                }
            }
        }
        
        results.insert(step.id.clone(), response_text.clone());
        
        memory_for_conditions.insert(step.id.clone(), response_text);
        
        current_step_idx += 1;
    }
    
    if is_interactive && !is_pipe && !json_output && config.interactive_config.save_path.is_some() {
        println!("💾 {}", "Interactive session complete. History saving will be available in a future version.".bright_blue());
    }
    
    if json_output {
        let final_result = if let Some(final_step) = config.steps.last() {
            results.get(&final_step.id).cloned().unwrap_or_default()
        } else {
            String::new()
        };
        
        let json_response = JsonResponse {
            chain_name: config.name.clone(),
            steps: results.clone(),
            result: final_result,
        };
        
        println!("{}", serde_json::to_string_pretty(&json_response)?);
    } else if is_pipe {
        if let Some(final_step) = config.steps.last() {
            if let Some(result) = results.get(&final_step.id) {
                println!("{}", result);
            }
        }
    } else {
        display_chain_results(&results, &config.steps);
    }
    
    Ok(())
}

/// Display the results of a chain execution
fn display_chain_results(results: &HashMap<String, String>, steps: &[StepConfig]) {
    let separator = "═".repeat(100);
    println!("\n{}", separator.bright_blue());
    println!("🔗 {}", "CHAIN EXECUTION RESULTS".bright_magenta().bold());
    println!("{}", separator.bright_blue());

    let mut results_table = Table::new();
    results_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(120);
    
    results_table.set_header(vec![
        Cell::new("STEP ID").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold),
        Cell::new("RESULT").fg(comfy_table::Color::Yellow).add_attribute(comfy_table::Attribute::Bold),
    ]);
    
    let mut has_results = false;
    
    for step in steps {
        if let Some(result) = results.get(&step.id) {
            has_results = true;
            
            let display_result = if result.len() > 500 {
                let wrapped = textwrap::fill(&result[0..497], 100);
                format!("{}...", wrapped)
            } else {
                textwrap::fill(result, 100)
            };
            
            results_table.add_row(vec![
                Cell::new(&step.id).fg(comfy_table::Color::Cyan).add_attribute(comfy_table::Attribute::Bold),
                Cell::new(&display_result),
            ]);
        }
    }
    
    if has_results {
        println!("{}", results_table);
        
        let completed_steps = results.len();
        let total_steps = steps.len();
        let completion_percentage = if total_steps > 0 {
            (completed_steps * 100) / total_steps
        } else {
            0
        };
        println!("\n📊 {}: {}/{} ({}%)", 
            "Completed steps".bright_yellow(), 
            completed_steps, 
            total_steps,
            completion_percentage
        );
    } else {
        println!("⚠️ {}", "No results to display".bright_red());
    }
}

/// Load a chain configuration from a file
fn load_chain_config(file_path: &PathBuf) -> Result<ChainConfig, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;
    
    if let Some(ext) = file_path.extension() {
        if ext == "json" {
            Ok(serde_json::from_str(&content)?)
        } else if ext == "yaml" || ext == "yml" {
            Ok(serde_yaml::from_str(&content)?)
        } else {
            Err(format!("Unsupported file format: {}", ext.to_string_lossy()).into())
        }
    } else {
        match serde_json::from_str(&content) {
            Ok(config) => Ok(config),
            Err(_) => Ok(serde_yaml::from_str(&content)?),
        }
    }
}

/// Create a template chain configuration file
fn create_chain_template(output: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let template = ChainConfig {
        name: "example-chain".to_string(),
        description: Some("A chain that demonstrates multi-step LLM processing with conditionals and interactive mode".to_string()),
        default_provider: Some("openai:gpt-4o".to_string()),
        input_var: "input".to_string(),
        interactive_config: InteractiveConfig {
            auto_start: false,
            default_steps: vec!["topic".to_string(), "library_details".to_string()],
            timeout: 300,
            save_path: None,
        },
        steps: vec![
            StepConfig {
                id: "topic".to_string(),
                template: "Suggest an interesting technical topic to explore based on this input: {{input}}. If no input is provided, choose something you think is interesting. Answer with just the topic name.".to_string(),
                provider_id: None,
                mode: "chat".to_string(),
                temperature: Some(0.7),
                max_tokens: Some(50),
                condition: None,
                interactive: true,
            },
            StepConfig {
                id: "details".to_string(),
                template: "List 3 key aspects of {{topic}} that developers should know. Format as bullet points.".to_string(),
                provider_id: None,
                mode: "chat".to_string(),
                temperature: Some(0.5),
                max_tokens: Some(200),
                condition: None,
                interactive: false,
            },
            StepConfig {
                id: "library_check".to_string(),
                template: "Based on the topic '{{topic}}', is there a popular library or framework that developers commonly use? Respond with just the library name, or 'none' if there isn't a clear one.".to_string(),
                provider_id: None,
                mode: "chat".to_string(),
                temperature: Some(0.3),
                max_tokens: Some(50),
                condition: None,
                interactive: false,
            },
            StepConfig {
                id: "library_details".to_string(),
                template: "Describe the key features and benefits of the {{library_check}} library for working with {{topic}}.".to_string(),
                provider_id: None,
                mode: "chat".to_string(),
                temperature: Some(0.3),
                max_tokens: Some(200),
                condition: Some("!library_check=none".to_string()),
                interactive: true,
            },
            StepConfig {
                id: "code_example".to_string(),
                template: "Based on {{topic}} and these aspects: {{details}}, provide a code example that demonstrates one of these aspects{{#library_check}} using the {{library_check}} library{{/library_check}}.".to_string(),
                provider_id: None,
                mode: "chat".to_string(),
                temperature: Some(0.3),
                max_tokens: Some(400),
                condition: None,
                interactive: false,
            },
            StepConfig {
                id: "system_info".to_string(),
                template: "This analysis was generated on {{sys.date}} at {{sys.time}} on a {{sys.os}} system.".to_string(),
                provider_id: None,
                mode: "chat".to_string(),
                temperature: Some(0.1),
                max_tokens: Some(50),
                condition: None,
                interactive: false,
            },
        ],
    };
    
    if let Some(ext) = output.extension() {
        if ext == "json" {
            fs::write(output, serde_json::to_string_pretty(&template)?)?;
        } else if ext == "yaml" || ext == "yml" {
            fs::write(output, serde_yaml::to_string(&template)?)?;
        } else {
            return Err(format!("Unsupported output format: {}", ext.to_string_lossy()).into());
        }
    } else {
        let mut output_with_ext = output.clone();
        output_with_ext.set_extension("yaml");
        fs::write(&output_with_ext, serde_yaml::to_string(&template)?)?;
    }
    
    println!("✅ Chain template created at {}", output.display());
    println!("Edit this file to customize your chain, then run with: llm-chain --file {}", output.display());
    
    Ok(())
}

/// Display available providers
fn display_providers() {
    let mut providers_table = Table::new();
    providers_table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_width(80);
    
    providers_table.add_row(vec![
        Cell::new("Available Providers").fg(comfy_table::Color::Cyan).add_attribute(comfy_table::Attribute::Bold),
    ]);
    
    providers_table.add_row(vec![Cell::new("OpenAI (openai)")]);
    providers_table.add_row(vec![Cell::new("Anthropic (anthropic)")]);
    providers_table.add_row(vec![Cell::new("Google (google)")]);
    providers_table.add_row(vec![Cell::new("Ollama (ollama)")]);
    providers_table.add_row(vec![Cell::new("DeepSeek (deepseek)")]);
    providers_table.add_row(vec![Cell::new("Groq (groq)")]);
    providers_table.add_row(vec![Cell::new("XAI (xai)")]);
    providers_table.add_row(vec![Cell::new("Phind (phind)")]);
    
    println!("{}", providers_table);
    
    println!("To use a provider in a chain, specify it as 'provider:model' in the chain configuration file.");
    println!("Example: 'openai:gpt-4o' or 'anthropic:claude-3-5-sonnet-20240620'");
}

/// Retrieves provider and model information from various sources
fn get_provider_info(args: &CliArgs, config: &ChainConfig) -> Result<(String, Option<String>), Box<dyn std::error::Error>> {
    if let Some(provider_string) = &args.provider {
        let parts: Vec<&str> = provider_string.split(':').collect();
        return Ok((parts[0].to_string(), parts.get(1).map(|s| s.to_string())));
    }
    
    if let Some(default_provider) = &config.default_provider {
        let parts: Vec<&str> = default_provider.split(':').collect();
        return Ok((parts[0].to_string(), parts.get(1).map(|s| s.to_string())));
    }
    
    if let Some(default_provider) = SecretStore::new().ok().and_then(|store| store.get_default_provider().cloned()) {
        let parts: Vec<&str> = default_provider.split(':').collect();
        return Ok((parts[0].to_string(), parts.get(1).map(|s| s.to_string())));
    }
    
    Err("No provider specified. Use --provider, or define default_provider in your chain configuration file.".into())
}

/// Retrieves the appropriate API key for the specified backend
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

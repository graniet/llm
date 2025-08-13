//! End-to-end example showing a realistic tool-use cycle:
//! 1. The user asks to import users.
//! 2. The model replies with a `tool_use` call.
//! 3. We execute the function on our side (mock).
//! 4. We send back a `tool_result` message.
//! 5. The model produces a final confirmation message.

use llm::builder::{FunctionBuilder, LLMBackend, LLMBuilder};
use llm::chat::{ChatMessage, ToolChoice};
use llm::{FunctionCall, ToolCall};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct User {
    name: String,
    emails: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ImportUsersArgs {
    users: Vec<User>,
}

fn import_users_tool() -> FunctionBuilder {
    let schema = json!({
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "emails": {
                            "type": "array",
                            "items": { "type": "string", "format": "email" }
                        }
                    },
                    "required": ["name", "emails"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["users"],
        "additionalProperties": false
    });

    FunctionBuilder::new("import_users")
        .description("Bulk-import a list of users with their email addresses.")
        .json_schema(schema)
}

fn import_users(args_json: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let args: ImportUsersArgs = serde_json::from_str(args_json)?;
    println!("[server] imported {} users", args.users.len());
    Ok(args.users.len())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .function(import_users_tool())
        .tool_choice(ToolChoice::Any)
        .build()?;

    let mut messages = vec![ChatMessage::user()
        .content("Please import Alice <alice@example.com> and Bob <bob@example.com>.")
        .build()];

    let first_resp = llm.chat(&messages).await?;
    println!("[assistant] {first_resp}");

    if let Some(tool_calls) = first_resp.tool_calls() {
        let mut tool_results = Vec::new();
        for call in &tool_calls {
            match import_users(&call.function.arguments) {
                Ok(count) => {
                    // Prepare a ToolResult conveying success.
                    tool_results.push(ToolCall {
                        id: call.id.clone(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: call.function.name.clone(),
                            arguments: json!({ "status": "ok", "imported": count }).to_string(),
                        },
                    });
                }
                Err(e) => {
                    eprintln!("[server] import failed: {e}");
                }
            }
        }

        messages.push(ChatMessage::assistant().tool_use(tool_calls).build());
        messages.push(ChatMessage::assistant().tool_result(tool_results).build());

        let final_resp = llm.chat(&messages).await?;
        println!("[assistant] {final_resp}");
    }

    Ok(())
}

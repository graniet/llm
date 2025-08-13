//! This example demonstrates how to list available models from an LLM backend.
//! It uses the `ModelListRequest` structure to filter models based on specific criteria.
//! The example demonstrates how to build an LLM client, create a model list request,
//! and handle the response from the LLM backend.

use llm::{
    builder::{LLMBackend, LLMBuilder},
    models::ModelListRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("sk-TESTKEY".into());

    let llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .build()
        .expect("Failed to build LLM (OpenAI)");

    let request: Option<&ModelListRequest> = None;

    match llm.list_models(request).await {
        Ok(response) => {
            println!("Models available for backend {:?}:", response.get_backend());
            for model_id in response.get_models() {
                println!("- {model_id}");
            }
        }
        Err(e) => eprintln!("Error listing models: {e}"),
    }

    Ok(())
}

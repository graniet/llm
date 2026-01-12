use llm::secret_store::SecretStore;

use crate::args::{CliArgs, CommandKind};
use crate::config::AppConfig;
use crate::model::ModelCatalog;
use crate::provider::{resolve_selection, ProviderFactory, ProviderOverrides, ProviderRegistry};

pub fn handle_command(kind: CommandKind, args: &CliArgs) -> anyhow::Result<()> {
    let mut store = SecretStore::new()?;
    match kind {
        CommandKind::Set => handle_set(&mut store, args)?,
        CommandKind::Get => handle_get(&store, args),
        CommandKind::Delete => handle_delete(&mut store, args)?,
        CommandKind::Default => handle_default(&mut store, args)?,
        CommandKind::Chat => (),
    }
    Ok(())
}

fn handle_set(store: &mut SecretStore, args: &CliArgs) -> anyhow::Result<()> {
    let key = args.provider_or_key.as_deref();
    let value = args.prompt_or_value.as_deref();
    if let (Some(key), Some(value)) = (key, value) {
        store.set(key, value)?;
        println!("Secret '{}' has been set.", key);
    }
    Ok(())
}

fn handle_get(store: &SecretStore, args: &CliArgs) {
    let Some(key) = args.provider_or_key.as_deref() else {
        return;
    };
    if let Some(value) = store.get(key) {
        println!("{key}: {value}");
    } else {
        println!("Secret '{}' not found", key);
    }
}

fn handle_delete(store: &mut SecretStore, args: &CliArgs) -> anyhow::Result<()> {
    if let Some(key) = args.provider_or_key.as_deref() {
        store.delete(key)?;
        println!("Secret '{}' deleted.", key);
    }
    Ok(())
}

fn handle_default(store: &mut SecretStore, args: &CliArgs) -> anyhow::Result<()> {
    if let Some(provider) = args.provider_or_key.as_deref() {
        store.set_default_provider(provider)?;
        println!("Default provider set to {provider}");
    } else if let Some(provider) = store.get_default_provider() {
        println!("Default provider: {provider}");
    } else {
        println!("No default provider set");
    }
    Ok(())
}

pub fn list_providers(registry: &ProviderRegistry) {
    for info in registry.list() {
        println!("{} ({})", info.display_name, info.id);
    }
}

pub async fn list_models(
    args: &CliArgs,
    config: &AppConfig,
    registry: &ProviderRegistry,
) -> anyhow::Result<()> {
    let default_provider = SecretStore::new()
        .ok()
        .and_then(|store| store.get_default_provider().cloned());
    let selection = resolve_selection(args, config, default_provider)?;
    let overrides = ProviderOverrides::default();
    let factory = ProviderFactory::new(config, registry);
    let provider = factory.build(&selection, overrides)?;
    let models = config
        .providers
        .get(provider.id.as_str())
        .map(|cfg| ModelCatalog::from_config(&cfg.models))
        .unwrap_or_else(|| ModelCatalog::from_config(&[]));
    let catalog = if models.is_empty() {
        ModelCatalog::from_provider(provider.provider.clone()).await
    } else {
        models
    };
    for model in catalog.list() {
        println!("{}", model.id);
    }
    Ok(())
}

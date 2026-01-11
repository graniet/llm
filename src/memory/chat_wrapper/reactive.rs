use std::sync::{atomic::AtomicU32, Arc};

use tokio::sync::{broadcast, RwLock};

use crate::{
    chat::{ChatMessage, ChatRole},
    memory::{MemoryProvider, MessageCondition, MessageEvent},
    LLMProvider,
};

pub(super) struct ReactiveConfig {
    memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
    provider: Arc<dyn LLMProvider>,
    role_triggers: Vec<(String, MessageCondition)>,
    role: Option<String>,
    max_cycles: Option<u32>,
    cycle_counter: Arc<AtomicU32>,
}

impl ReactiveConfig {
    pub(super) fn new(
        memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
        provider: Arc<dyn LLMProvider>,
    ) -> Self {
        Self {
            memory,
            provider,
            role_triggers: Vec::new(),
            role: None,
            max_cycles: None,
            cycle_counter: Arc::new(AtomicU32::new(0)),
        }
    }

    pub(super) fn role_triggers(mut self, triggers: Vec<(String, MessageCondition)>) -> Self {
        self.role_triggers = triggers;
        self
    }

    pub(super) fn role(mut self, role: Option<String>) -> Self {
        self.role = role;
        self
    }

    pub(super) fn max_cycles(mut self, max_cycles: Option<u32>) -> Self {
        self.max_cycles = max_cycles;
        self
    }

    pub(super) fn cycle_counter(mut self, counter: Arc<AtomicU32>) -> Self {
        self.cycle_counter = counter;
        self
    }
}

pub(super) fn spawn_reactive_listener(config: ReactiveConfig) {
    tokio::spawn(async move {
        run_reactive(config).await;
    });
}

async fn run_reactive(config: ReactiveConfig) {
    let mut receiver = match get_receiver(&config).await {
        Some(receiver) => receiver,
        None => return,
    };

    while let Ok(event) = receiver.recv().await {
        if !should_handle(&config, &event) || reached_max(&config) {
            continue;
        }
        let context = load_context(&config).await;
        if let Some(text) = generate_reply(&config, &context).await {
            store_reply(&config, text).await;
        }
    }
}

async fn get_receiver(config: &ReactiveConfig) -> Option<broadcast::Receiver<MessageEvent>> {
    let guard = config.memory.read().await;
    guard.get_event_receiver()
}

fn should_handle(config: &ReactiveConfig, event: &MessageEvent) -> bool {
    let event_role = resolve_event_role(event);
    config
        .role_triggers
        .iter()
        .any(|(role, cond)| role == event_role && cond.matches(event))
}

fn resolve_event_role(event: &MessageEvent) -> &str {
    if event.msg.role == ChatRole::User {
        "user"
    } else {
        &event.role
    }
}

fn reached_max(config: &ReactiveConfig) -> bool {
    let Some(max) = config.max_cycles else {
        return false;
    };
    config
        .cycle_counter
        .load(std::sync::atomic::Ordering::Relaxed)
        >= max
}

async fn load_context(config: &ReactiveConfig) -> Vec<ChatMessage> {
    let guard = config.memory.read().await;
    match guard.recall("", None).await {
        Ok(messages) => messages,
        Err(err) => {
            log::warn!("Reactive memory recall error: {err}");
            Vec::new()
        }
    }
}

async fn generate_reply(config: &ReactiveConfig, context: &[ChatMessage]) -> Option<String> {
    let response = match config.provider.chat(context).await {
        Ok(response) => response,
        Err(err) => {
            log::warn!("Reactive chat error: {err}");
            return None;
        }
    };
    response.text()
}

async fn store_reply(config: &ReactiveConfig, text: String) {
    let Some(role) = config.role.clone() else {
        return;
    };
    let msg = ChatMessage::assistant()
        .content(format!("[{role}] {text}"))
        .build();
    let mut guard = config.memory.write().await;
    if let Err(err) = guard.remember_with_role(&msg, role).await {
        log::warn!("Reactive memory save error: {err}");
        return;
    }
    config
        .cycle_counter
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

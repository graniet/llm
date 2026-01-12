use super::super::AppController;

pub fn start_new_conversation(controller: &mut AppController) -> bool {
    let provider_id = controller
        .state
        .active_conversation()
        .map(|conv| conv.provider_id.clone())
        .or_else(|| {
            controller
                .state
                .config
                .default_provider
                .clone()
                .map(Into::into)
        })
        .unwrap_or_else(|| "openai".into());
    controller.state.conversations.new_conversation(
        provider_id,
        controller.state.config.default_model.clone(),
        controller.state.config.chat.system_prompt.clone(),
    );
    controller.state.scroll.reset();
    controller.record_snapshot();
    true
}

pub fn fork_conversation(controller: &mut AppController) -> bool {
    let id = match controller.state.active_conversation_id() {
        Some(id) => id,
        None => return false,
    };
    controller.state.conversations.fork_conversation(id);
    controller.state.scroll.reset();
    controller.record_snapshot();
    true
}

pub fn save_active_conversation(controller: &mut AppController) -> bool {
    if let Some(conv) = controller.state.active_conversation() {
        let _ = controller.state.store.save(conv);
        return true;
    }
    false
}

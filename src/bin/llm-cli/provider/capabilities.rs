use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub streaming: bool,
    pub tools: bool,
    pub tool_streaming: bool,
    pub vision: bool,
    pub models_list: bool,
}

impl ProviderCapabilities {
    pub const FULL: Self = Self {
        streaming: true,
        tools: true,
        tool_streaming: true,
        vision: true,
        models_list: true,
    };
    pub const TOOLS_NO_STREAM: Self = Self {
        streaming: true,
        tools: true,
        tool_streaming: false,
        vision: true,
        models_list: true,
    };
    pub const LOCAL_BASIC: Self = Self {
        streaming: true,
        tools: false,
        tool_streaming: false,
        vision: true,
        models_list: false,
    };
    pub const STREAM_ONLY: Self = Self {
        streaming: true,
        tools: false,
        tool_streaming: false,
        vision: false,
        models_list: false,
    };
    pub const NONE: Self = Self {
        streaming: false,
        tools: false,
        tool_streaming: false,
        vision: false,
        models_list: false,
    };
}

mod animation;
mod collapsible;
mod context;
pub mod controller;
mod events;
mod overlay;
mod paste;
mod runner;
mod scroll;
mod state;
mod status;
mod streaming;
mod terminal;

pub use crate::history::SnapshotId;
pub use animation::AnimationState;
pub use collapsible::{CollapsibleState, TOOL_COLLAPSE_LINES};
pub use context::{
    compact_conversation, context_limit, summarize_conversation_head, usage_for, ContextUsage,
};
pub use events::{AppEvent, InputEvent, StreamEvent, ToolApprovalRequest, ToolEvent};
pub use overlay::{
    BacktrackOverlayState, ConfirmState, DiffViewerState, OnboardingProvider, OnboardingState,
    OnboardingStep, OverlayState, PagerState, PickerItem, PickerState, SearchState, SlashCommand,
    SlashCommandId, SlashCommandState, ToolApprovalState, ToolBuilderResult, ToolBuilderState,
    ToolBuilderStep, UserToolDraft,
};
pub use paste::PasteDetector;
pub use runner::run_app;
pub use scroll::ScrollState;
pub use state::AppState;
pub use state::{Focus, InputMode};
pub use status::AppStatus;
pub use status::StatusMetrics;
pub use streaming::StreamManager;
pub use terminal::{init_terminal, restore_terminal, AppTerminal};

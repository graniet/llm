mod backtrack;
mod confirm;
mod dialogue_builder;
mod diff;
mod onboarding;
mod pager;
mod picker;
mod search;
mod slash;
mod tool_approval;
mod tool_builder;

pub use backtrack::BacktrackOverlayState;
pub use confirm::ConfirmState;
pub use dialogue_builder::{
    DialogueBuilderResult, DialogueBuilderState, DialogueBuilderStep, ParticipantField,
};
pub use diff::DiffViewerState;
pub use onboarding::{OnboardingProvider, OnboardingState, OnboardingStep};
pub use pager::PagerState;
pub use picker::{PickerItem, PickerState};
pub use search::SearchState;
pub use slash::{SlashCommand, SlashCommandId, SlashCommandState};
pub use tool_approval::ToolApprovalState;
pub use tool_builder::{ToolBuilderResult, ToolBuilderState, ToolBuilderStep, UserToolDraft};

#[derive(Debug)]
pub enum OverlayState {
    None,
    Help,
    ProviderPicker(PickerState),
    ModelPicker(PickerState),
    ConversationPicker(PickerState),
    SkillPicker(PickerState),
    ToolPicker(PickerState),
    Onboarding(OnboardingState),
    Pager(PagerState),
    Backtrack(BacktrackOverlayState),
    DiffViewer(DiffViewerState),
    SlashCommands(SlashCommandState),
    ConfirmExit(ConfirmState),
    ToolApproval(ToolApprovalState),
    ToolBuilder(ToolBuilderState),
    DialogueBuilder(DialogueBuilderState),
    Search(SearchState),
}

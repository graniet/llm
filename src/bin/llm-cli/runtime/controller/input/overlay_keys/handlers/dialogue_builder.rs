use crossterm::event::{KeyCode, KeyEvent};

use crate::runtime::overlay::{DialogueBuilderResult, DialogueBuilderState, DialogueBuilderStep};

use super::super::{OverlayAction, OverlayResult};

pub(super) fn handle_dialogue_builder(
    state: &mut DialogueBuilderState,
    key: KeyEvent,
) -> OverlayResult {
    // Special handling for ConfigureParticipant step
    if state.step == DialogueBuilderStep::ConfigureParticipant {
        return handle_configure_participant(state, key);
    }

    match key.code {
        KeyCode::Esc => {
            // Go back to previous step or close
            if state.prev_step() {
                OverlayResult::action(OverlayAction::Handled)
            } else {
                OverlayResult::close(OverlayAction::None)
            }
        }
        KeyCode::Enter => handle_enter(state),
        KeyCode::Backspace => handle_backspace(state),
        KeyCode::Char(c) => handle_char(state, c),
        KeyCode::Up => {
            state.prev();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Down => {
            state.next();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Tab => {
            // Move to next step if validation passes
            if let Err(err) = state.validate() {
                state.set_error(err);
                OverlayResult::action(OverlayAction::Handled)
            } else if state.next_step() {
                OverlayResult::action(OverlayAction::Handled)
            } else {
                // Already at last step, complete
                OverlayResult::close(OverlayAction::DialogueBuilderComplete(
                    DialogueBuilderResult::from(&*state),
                ))
            }
        }
        KeyCode::BackTab => {
            state.prev_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Delete => {
            if state.step == DialogueBuilderStep::Participants {
                state.remove_participant();
            }
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}

fn handle_configure_participant(state: &mut DialogueBuilderState, key: KeyEvent) -> OverlayResult {
    match key.code {
        KeyCode::Esc => {
            // Cancel and go back without saving
            state.step = DialogueBuilderStep::Participants;
            state.input.clear();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Enter => {
            // Save and go back
            state.finish_configure();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Tab => {
            // Switch to next field
            state.next_field();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::BackTab => {
            // Switch to previous field (same as next since only 2 fields)
            state.next_field();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Backspace => {
            state.pop_input();
            OverlayResult::action(OverlayAction::Handled)
        }
        KeyCode::Char(c) => {
            state.push_input(c);
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}

fn handle_enter(state: &mut DialogueBuilderState) -> OverlayResult {
    match state.step {
        DialogueBuilderStep::Participants => {
            // Add participant from input
            if !state.input.is_empty() {
                state.add_participant();
            }
            OverlayResult::action(OverlayAction::Handled)
        }
        DialogueBuilderStep::ConfigureParticipant => {
            // Handled by handle_configure_participant, but add for exhaustiveness
            state.finish_configure();
            OverlayResult::action(OverlayAction::Handled)
        }
        DialogueBuilderStep::Mode => {
            // Select current mode
            state.select_mode();
            OverlayResult::action(OverlayAction::Handled)
        }
        DialogueBuilderStep::InitialPrompt => {
            // Accept prompt and move to next step
            state.initial_prompt = state.input.clone();
            state.input.clear();
            state.next_step();
            OverlayResult::action(OverlayAction::Handled)
        }
        DialogueBuilderStep::Review => {
            // Complete
            if state.is_complete() {
                OverlayResult::close(OverlayAction::DialogueBuilderComplete(
                    DialogueBuilderResult::from(&*state),
                ))
            } else {
                state.set_error("At least 2 participants required");
                OverlayResult::action(OverlayAction::Handled)
            }
        }
    }
}

fn handle_backspace(state: &mut DialogueBuilderState) -> OverlayResult {
    state.pop_input();
    OverlayResult::action(OverlayAction::Handled)
}

fn handle_char(state: &mut DialogueBuilderState, c: char) -> OverlayResult {
    match state.step {
        DialogueBuilderStep::Participants => {
            // 'e' to configure selected participant (when input is empty)
            if c == 'e' && state.input.is_empty() && state.configure_participant() {
                return OverlayResult::action(OverlayAction::Handled);
            }
            state.push_input(c);
            OverlayResult::action(OverlayAction::Handled)
        }
        DialogueBuilderStep::InitialPrompt => {
            state.push_input(c);
            OverlayResult::action(OverlayAction::Handled)
        }
        _ => OverlayResult::action(OverlayAction::None),
    }
}

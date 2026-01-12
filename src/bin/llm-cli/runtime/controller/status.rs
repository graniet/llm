use crate::runtime::AppStatus;

use super::AppController;

impl AppController {
    pub fn set_status(&mut self, status: AppStatus) {
        let was_busy = self.state.status.is_busy();
        let is_busy = status.is_busy();
        if is_busy && !was_busy {
            self.state.status_metrics.start();
        }
        if !is_busy {
            self.state.status_metrics.stop();
        }
        self.state.status = status;
    }

    pub fn update_status_tokens(&mut self, tokens: u32) {
        self.state.status_metrics.update_tokens(tokens);
    }
}

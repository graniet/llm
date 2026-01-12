use crate::runtime::{OverlayState, PagerState};

use super::AppController;

impl AppController {
    pub fn open_config_overlay(&mut self) -> bool {
        let mut body = match toml::to_string_pretty(&self.state.config) {
            Ok(value) => value,
            Err(err) => {
                self.set_status(crate::runtime::AppStatus::Error(format!(
                    "config format: {err}"
                )));
                return false;
            }
        };
        let header = format!(
            "Config: {}\nSecrets: ~/.llm/secrets.json\n\n",
            self.config_paths.config_file.display()
        );
        body.insert_str(0, &header);
        self.state.overlay = OverlayState::Pager(PagerState::new("Config", &body));
        true
    }
}

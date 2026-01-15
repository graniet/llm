use super::super::AppController;

const PAGE_SCROLL_LINES: u16 = 10;

pub fn scroll_up(controller: &mut AppController, lines: u16) -> bool {
    controller.state.scroll.scroll_up(lines);
    true
}

pub fn scroll_down(controller: &mut AppController, lines: u16) -> bool {
    controller.state.scroll.scroll_down(lines);
    true
}

pub fn scroll_to_top(controller: &mut AppController) -> bool {
    controller.state.scroll.scroll_up(u16::MAX);
    true
}

pub fn scroll_to_bottom(controller: &mut AppController) -> bool {
    controller.state.scroll.reset();
    true
}

pub fn clear_screen(controller: &mut AppController) -> bool {
    controller.state.scroll.reset();
    true
}

pub fn page_up(controller: &mut AppController) -> bool {
    controller.state.scroll.scroll_up(PAGE_SCROLL_LINES);
    true
}

pub fn page_down(controller: &mut AppController) -> bool {
    controller.state.scroll.scroll_down(PAGE_SCROLL_LINES);
    true
}

pub(super) const SCROLL_STEP: u16 = 1;
const PAGER_PADDING: u16 = 4;

pub(super) fn pager_height(height: u16) -> u16 {
    height.saturating_sub(PAGER_PADDING).max(1)
}

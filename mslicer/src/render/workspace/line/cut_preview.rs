use nalgebra::Vector3;

use crate::{
    core::App,
    render::workspace::line::{Line, LineGenerator},
};

/// Outline of the cut and of the cut plane, while the Cut section of the
/// model panel is open.
pub struct CutPreviewDispatch {
    cached_lines: Vec<Line>,
}

impl CutPreviewDispatch {
    pub fn new() -> Self {
        Self {
            cached_lines: Vec::new(),
        }
    }
}

impl LineGenerator for CutPreviewDispatch {
    fn generate_lines(&mut self, app: &mut App) {
        self.cached_lines.clear();
        if app.state.cut.visible {
            let color = Vector3::new(1.0, 0.85, 0.1);
            let lines = app.state.cut.preview.iter();
            (self.cached_lines).extend(lines.map(|[a, b]| Line::new(*a, *b).color(color)));
        }
    }

    fn lines(&self) -> &[Line] {
        &self.cached_lines
    }
}

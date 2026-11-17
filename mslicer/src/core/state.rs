use std::{sync::Arc, time::Instant};

use common::color::SRgb;
use egui::Vec2;
use egui_phosphor::regular::{CURSOR_CLICK, DROP_HALF_BOTTOM, LINE_SEGMENT, TRIANGLE};
use egui_tracing::EventCollector;
use nalgebra::{Matrix4, Vector2, Vector3};
use slicer::{geometry::mesh::Mesh, post_process::hollowing::DrainHole};
use tools::cut::TenonConfig;
use wgpu::Device;

use crate::{
    core::{
        config::peripherals::Webhook,
        project::model::{Model, ModelId},
        selected::{SelectedModel, SelectedPrinter, SelectedSupports},
    },
    interface::{keybinds::Keybind, tools::Tools},
    render::util::MeshBuffers,
    util::misc::DefaultNum,
};

// todo: organize this...
#[derive(Default)]
pub struct UiState {
    pub event_collector: EventCollector,
    pub close_requested: bool,
    pub queue_reset_ui: bool,
    pub line_support_debug: Vec<[Vector3<f32>; 2]>,
    pub tool: Tool,

    // support stuff
    pub workspace: WorkspaceHover,
    pub hovered_geometry: Option<GeometryHit>,

    // selections
    pub select_start: Option<Vec2>,
    pub selected: SelectedModel,
    pub selected_printer: SelectedPrinter,
    pub selected_supports: SelectedSupports,
    pub support_preview: Option<Mesh>,
    pub drain_hole: DrainHoleTool,
    pub cut: CutTool,
    pub selected_remap_point: Option<u8>,

    // remote send ui
    pub working_address: String,
    pub working_filename: String,
    pub remote_print_connecting: RemotePrintConnectStatus,
    pub shared_webhook: Arc<SharedPrintCompletion>,

    // slice preview
    pub preview_layer: usize,
    pub last_preview_layer: usize,
    pub preview_offset: Vector2<f32>,
    pub preview_scale: f32,
    pub layer_count: (usize, u8),

    pub preview_mode: PreviewMode,
    pub last_preview_interact: Option<Instant>,

    pub reconstruct_subsample: DefaultNum<20, u8>,

    pub anisotropic_aa: bool,

    pub tools: Tools,
    pub move_timeout: u32, // todo: having this in frames is fragile!
}

pub struct PreviewModel {
    pub mesh: MeshBuffers,
    pub faces: usize,

    pub transform: Matrix4<f32>,
    pub color: SRgb<f32>,
}

#[derive(Default, Copy, Clone, PartialEq, Eq)]
pub enum PreviewMode {
    Interactive,
    #[default]
    Static,
}

#[derive(Default, PartialEq, Eq)]
pub enum Tool {
    #[default]
    Select,
    Support,
    Orient,
    DrainHole,
}

pub struct DrainHoleTool {
    /// Diameter of newly placed drain holes, in mm.
    pub diameter: f32,
    /// The hole that clicking would place, on the hovered model.
    pub preview: Option<(ModelId, DrainHole)>,
    /// The existing hole (model, index) hovered while Ctrl is held, that
    /// clicking would remove.
    pub hovered: Option<(ModelId, usize)>,
}

/// Settings of the Cut section of the model panel.
#[derive(Default)]
pub struct CutTool {
    pub axis: CutAxis,
    /// Position of the plane across the model along the axis (0 to 1), so it
    /// follows the model when it is moved or scaled.
    pub fraction: f32,
    /// Puts the pegs on the part with the higher coordinates.
    pub flip: bool,
    pub tenons: TenonConfig,

    /// The model `fraction` was set for.
    pub model: Option<ModelId>,
    /// Whether the cut settings are shown this frame, which shows the preview.
    pub visible: bool,
    /// World space lines of the cut outline and plane.
    pub preview: Vec<[Vector3<f32>; 2]>,
    pub preview_key: Option<(ModelId, Matrix4<f32>, CutAxis, Vector3<f32>)>,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutAxis {
    X,
    Y,
    #[default]
    Z,
}

impl CutAxis {
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    pub fn vector(&self) -> Vector3<f32> {
        match self {
            Self::X => Vector3::x(),
            Self::Y => Vector3::y(),
            Self::Z => Vector3::z(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
        }
    }

    /// Names of the lower and higher side of the plane.
    pub fn side_names(&self) -> [&'static str; 2] {
        match self {
            Self::X => ["Left Part", "Right Part"],
            Self::Y => ["Front Part", "Back Part"],
            Self::Z => ["Bottom Part", "Top Part"],
        }
    }
}

impl Default for DrainHoleTool {
    fn default() -> Self {
        Self {
            diameter: 3.0,
            preview: None,
            hovered: None,
        }
    }
}

#[derive(Default)]
pub struct SharedPrintCompletion {
    pub webhook: Webhook,
    pub alert: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct GeometryHit {
    pub model: ModelId,
    pub support: bool,

    pub face: u32,
}

#[derive(Default)]
pub struct WorkspaceHover {
    pub is_moving: bool,
    pub aspect: f32,
    pub uv: Vector2<f32>,
}

#[derive(Default, PartialEq, Eq)]
pub enum RemotePrintConnectStatus {
    #[default]
    None,
    Connecting,
    Scanning,
}

impl PreviewModel {
    pub fn for_model(device: &Device, model: &mut Model) -> Self {
        Self {
            mesh: model.get_buffers(device).clone(),
            faces: model.mesh.face_count(),

            transform: *model.mesh.transformation_matrix(),
            color: model.color.to_srgb(),
        }
    }
}

impl WorkspaceHover {
    pub fn new(is_moving: bool, aspect: f32, uv: Vec2) -> Self {
        Self {
            is_moving,
            aspect,
            uv: Vector2::new(uv.x, uv.y),
        }
    }

    pub fn hovered(&self) -> bool {
        self.uv.x >= 0.0 && self.uv.y >= 0.0
    }
}

impl PreviewMode {
    pub fn other(&self) -> Self {
        match self {
            PreviewMode::Interactive => PreviewMode::Static,
            PreviewMode::Static => PreviewMode::Interactive,
        }
    }
}

impl Tool {
    pub const ALL: [Self; 4] = [Self::Select, Self::Support, Self::Orient, Self::DrainHole];

    pub fn keybind(&self) -> Keybind {
        match self {
            Tool::Select => Keybind::Pointer,
            Tool::Support => Keybind::Support,
            Tool::Orient => Keybind::Orient,
            Tool::DrainHole => Keybind::DrainHole,
        }
    }

    pub fn icon(&self) -> &str {
        match self {
            Tool::Select => CURSOR_CLICK,
            Tool::Support => LINE_SEGMENT,
            Tool::Orient => TRIANGLE,
            Tool::DrainHole => DROP_HALF_BOTTOM,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Tool::Select => "Select (Default)",
            Tool::Support => "Support Placement",
            Tool::Orient => "Orient to Face",
            Tool::DrainHole => "Drain Hole Placement\nCtrl + click on a hole to remove it",
        }
    }
}

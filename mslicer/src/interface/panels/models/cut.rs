//! Cut section of the model panel: cuts the model in two with a plane along
//! one axis, optionally with tenons so both parts can be glued back aligned.

use const_format::concatcp;
use egui::{ComboBox, DragValue, Ui, Widget};
use egui_phosphor::regular::{INFO, SCISSORS};
use nalgebra::Vector3;
use slicer::{
    geometry::{bounds::Bounds3D, mesh::Mesh},
    post_process::hollowing::DrainHole,
};
use tools::cut::{CutPart, CutPlane, cut};

use crate::{
    core::{
        App,
        history::{Action, ModelAction},
        project::model::Model,
        state::CutAxis,
    },
    interface::{
        components::grid,
        popup::{Popup, PopupIcon},
    },
    task::MeshAnalysis,
};

const CUT_TIP: &str = "Cuts the model in two along the plane shown in the viewport. The original model is hidden, not deleted.";
const TENON_TIP: &str = "Pegs on one part and matching pockets in the other, so both parts can be glued back perfectly aligned. They are placed automatically where the cut face is wide enough.";
const CLEARANCE_TIP: &str = "Gap between a peg and its pocket, on each side. Increase it if the parts are too tight to assemble.";
/// Distance between both parts after cutting, on top of the peg length (mm).
const EXPLODE_GAP: f32 = 5.0;

pub fn cut_section(app: &mut App, ui: &mut Ui, index: usize) {
    let model = &app.project.models[index];
    let (id, bounds) = (model.id, model.bounds.clone());

    let tool = &mut app.state.cut;
    if tool.model != Some(id) {
        tool.model = Some(id);
        tool.fraction = 0.5;
    }

    let mut cut_clicked = false;
    let response = ui.collapsing("Cut", |ui| {
        grid("model_cut_grid").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Axis");
                ui.label(INFO).on_hover_text(CUT_TIP);
            });
            ui.horizontal(|ui| {
                for axis in CutAxis::ALL {
                    let selected = tool.axis == axis;
                    if ui.selectable_label(selected, axis.name()).clicked() && !selected {
                        tool.axis = axis;
                        tool.fraction = 0.5;
                    }
                }
            });
            ui.end_row();

            ui.label("Position");
            let [low, high] = axis_range(&bounds, tool.axis);
            let mut position = low + tool.fraction * (high - low);
            let response = DragValue::new(&mut position)
                .range(low..=high)
                .speed(0.1)
                .max_decimals(2)
                .suffix(" mm")
                .ui(ui);
            if response.changed() && high > low {
                tool.fraction = (position - low) / (high - low);
            }
            ui.end_row();

            ui.horizontal(|ui| {
                ui.label("Tenons");
                ui.label(INFO).on_hover_text(TENON_TIP);
            });
            ui.checkbox(&mut tool.tenons.enabled, "");
            ui.end_row();

            let enabled = tool.tenons.enabled;
            ui.label("Pegs On");
            ui.add_enabled_ui(enabled, |ui| {
                let names = tool.axis.side_names();
                ComboBox::from_id_salt("cut_pegs_side")
                    .selected_text(names[tool.flip as usize])
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut tool.flip, false, names[0]);
                        ui.selectable_value(&mut tool.flip, true, names[1]);
                    });
            });
            ui.end_row();

            let tenons = &mut tool.tenons;
            let field = |ui: &mut Ui, label: &str, value: &mut f32, max: f32, speed: f64| {
                ui.label(label);
                ui.add_enabled(
                    enabled,
                    DragValue::new(value)
                        .range(0.0..=max)
                        .speed(speed)
                        .max_decimals(2)
                        .suffix(" mm"),
                );
                ui.end_row();
            };
            ui.label("Count");
            ui.add_enabled(enabled, DragValue::new(&mut tenons.count).range(1..=6))
                .on_hover_text("Maximum number of tenons on each separate cut face.");
            ui.end_row();
            field(ui, "Diameter", &mut tenons.diameter, 20.0, 0.05);
            field(ui, "Length", &mut tenons.height, 30.0, 0.05);
            tenons.diameter = tenons.diameter.max(1.0);
            tenons.height = tenons.height.max(1.0);

            ui.horizontal(|ui| {
                ui.label("Clearance");
                ui.label(INFO).on_hover_text(CLEARANCE_TIP);
            });
            ui.add_enabled(
                enabled,
                DragValue::new(&mut tenons.clearance)
                    .range(0.0..=1.0)
                    .speed(0.005)
                    .max_decimals(3)
                    .suffix(" mm"),
            );
            ui.end_row();
        });

        cut_clicked = ui.button(concatcp!(SCISSORS, " Cut")).clicked();
    });

    app.state.cut.visible = response.body_returned.is_some();
    if app.state.cut.visible {
        update_preview(app, index);
    }
    if cut_clicked {
        perform_cut(app, index);
    }
}

/// World space plane for the current settings. Its normal points to the part
/// receiving the pockets.
fn plane(app: &App, bounds: &Bounds3D) -> CutPlane {
    let tool = &app.state.cut;
    let axis = tool.axis.vector();
    let [low, high] = axis_range(bounds, tool.axis);
    CutPlane {
        point: axis * (low + tool.fraction * (high - low)),
        normal: if tool.flip { -axis } else { axis },
    }
}

fn axis_range(bounds: &Bounds3D, axis: CutAxis) -> [f32; 2] {
    let axis = axis.vector();
    [bounds.min.dot(&axis), bounds.max.dot(&axis)]
}

/// Lines of the cut outline on the model and of the plane around it.
fn update_preview(app: &mut App, index: usize) {
    let model = &app.project.models[index];
    let plane = plane(app, &model.bounds);
    let key = (
        model.id,
        *model.mesh.transformation_matrix(),
        app.state.cut.axis,
        plane.point,
    );
    if app.state.cut.preview_key == Some(key) {
        return;
    }

    let mesh = &model.mesh;
    let vertices = (mesh.vertices().iter())
        .map(|v| mesh.transform(v))
        .collect::<Vec<_>>();
    let distance = |v: Vector3<f32>| (v - plane.point).dot(&plane.normal);

    let mut lines = Vec::new();
    for face in mesh.faces() {
        let points = face.map(|i| vertices[i as usize]);
        let mut crossing = Vec::with_capacity(2);
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let (da, db) = (distance(points[a]), distance(points[b]));
            if (da > 0.0) != (db > 0.0) {
                crossing.push(points[a] + (points[b] - points[a]) * (da / (da - db)));
            }
        }
        if let [a, b] = crossing[..] {
            lines.push([a, b]);
        }
    }

    // Outline of the plane, a bit larger than the model.
    let bounds = &model.bounds;
    let margin = Vector3::repeat(3.0);
    let (min, max) = (bounds.min - margin, bounds.max + margin);
    let axis = app.state.cut.axis.vector();
    let flatten = |v: Vector3<f32>| v - axis * v.dot(&axis) + plane.point;
    let corners = match app.state.cut.axis {
        CutAxis::X => [
            (min.y, min.z),
            (max.y, min.z),
            (max.y, max.z),
            (min.y, max.z),
        ]
        .map(|(y, z)| Vector3::new(0.0, y, z)),
        CutAxis::Y => [
            (min.x, min.z),
            (max.x, min.z),
            (max.x, max.z),
            (min.x, max.z),
        ]
        .map(|(x, z)| Vector3::new(x, 0.0, z)),
        CutAxis::Z => [
            (min.x, min.y),
            (max.x, min.y),
            (max.x, max.y),
            (min.x, max.y),
        ]
        .map(|(x, y)| Vector3::new(x, y, 0.0)),
    }
    .map(flatten);
    for i in 0..4 {
        lines.push([corners[i], corners[(i + 1) % 4]]);
    }

    app.state.cut.preview = lines;
    app.state.cut.preview_key = Some(key);
}

fn perform_cut(app: &mut App, index: usize) {
    let plane = plane(app, &app.project.models[index].bounds);
    let tenons = app.state.cut.tenons;
    let platform = app.project.slice_config.platform_size;

    let original = &app.project.models[index];
    let mesh = &original.mesh;
    let vertices = (mesh.vertices().iter())
        .map(|v| mesh.transform(v))
        .collect::<Vec<_>>();
    let Some(result) = cut(&vertices, mesh.faces(), plane, &tenons) else {
        let body = "The cut plane does not cross this model. Move it within the model.";
        app.popup
            .open(Popup::simple("Nothing to Cut", PopupIcon::Error, body));
        return;
    };

    // Drain holes go to the part on their side of the plane.
    let holes = (original.drain_holes.iter())
        .map(|hole| {
            let position = mesh.transform(&hole.position);
            let normal = mesh.transform_normal(&hole.normal).normalize();
            let above = (position - plane.point).dot(&plane.normal) > 0.0;
            (above, position, normal, hole.radius)
        })
        .collect::<Vec<_>>();

    // Move the part with the higher coordinates away, without ever pushing a
    // part under the build plate.
    let gap = if result.tenons > 0 {
        tenons.height
    } else {
        0.0
    } + EXPLODE_GAP;
    let axis = app.state.cut.axis.vector();
    let above_is_higher = plane.normal.dot(&axis) > 0.0;

    let with_tenons = result.tenons > 0;
    let names = match with_tenons {
        true => ["pegs", "pockets"],
        false => ["1", "2"],
    };

    let mut parts = Vec::new();
    for (i, part) in [result.below, result.above].into_iter().enumerate() {
        let above = i == 1;
        let offset = if above == above_is_higher {
            axis * gap
        } else {
            Vector3::zeros()
        };
        let (mesh, center) = part_mesh(part, offset);

        let drain_holes = (holes.iter())
            .filter(|hole| hole.0 == above)
            .map(|&(_, position, normal, radius)| DrainHole {
                position: position - center,
                normal,
                radius,
            })
            .collect();

        let mut model = Model::from_mesh(mesh)
            .with_name(format!("{} ({})", original.name, names[i]))
            .with_color(original.color)
            .with_exposure(original.exposure)
            .with_hollow(original.hollow)
            .with_drain_holes(drain_holes)
            .with_collection(original.collection);
        if above {
            model = model.with_random_color();
        }
        model.update_oob(&platform);
        parts.push(model);
    }

    let original = &mut app.project.models[index];
    let original_id = original.id;
    app.history
        .track_model(original_id, ModelAction::Hidden(original.hidden));
    original.hidden = true;

    app.state.selected.clear();
    for (i, model) in parts.into_iter().enumerate() {
        app.history.track(Action::ModelAdded { id: model.id });
        app.tasks.add(MeshAnalysis::new(&model));
        (i == 0).then(|| app.state.selected.select_model(model.id));
        app.project.models.insert(index + 1 + i, model);
    }
    app.state.cut.model = None;

    let mut warnings = Vec::new();
    if result.open_outlines > 0 {
        warnings.push(format!(
            "The mesh has holes: {} cut outline(s) could not be closed. Repair the mesh (Mesh menu of the model) and cut again.",
            result.open_outlines
        ));
    }
    if tenons.enabled && result.faces_without_tenons > 0 {
        let (missing, total) = (result.faces_without_tenons, result.cut_faces);
        let summary = if missing == total {
            format!(
                "None of the {total} cut face(s) is large enough for {} mm tenons.",
                tenons.diameter
            )
        } else {
            format!(
                "Tenons were placed on {} of the {total} cut faces. The other {missing} are too small, or have too little material behind them, for {} mm tenons.",
                total - missing,
                tenons.diameter
            )
        };
        warnings.push(format!(
            "{summary} Parts without tenons have to be aligned by hand when gluing. Use smaller tenons or move the cut to add some."
        ));
    }
    if !warnings.is_empty() {
        let body = warnings.join("\n\n");
        app.popup
            .open(Popup::simple("Cut", PopupIcon::Warning, body));
    }
}

/// Builds a mesh from a cut part, centered on its bounding box and placed at
/// the same spot (plus `offset`).
fn part_mesh(part: CutPart, offset: Vector3<f32>) -> (Mesh, Vector3<f32>) {
    let bounds = Bounds3D::from_verts(&part.vertices);
    let center = (bounds.min + bounds.max) / 2.0;
    let vertices = part.vertices.into_iter().map(|v| v - center).collect();

    let mut mesh = Mesh::new(vertices, part.faces);
    mesh.set_position(center + offset);
    (mesh, center)
}

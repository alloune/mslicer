use common::{
    misc::{separate_thousands, subscript_number},
    units::{Centimeter, Milimeter, Milimeters},
};
use const_format::concatcp;
use egui::{
    Align, CollapsingHeader, Color32, ComboBox, Context, DragValue, Label, Layout, Popup, Response,
    Sense, TextEdit, Ui, UiBuilder, Widget, text::CCursorRange, vec2,
};
use egui_phosphor::regular::{
    ARROW_LINE_DOWN, ARROWS_COUNTER_CLOCKWISE, BANDAIDS, COPY, CUBE, CURSOR_TEXT, DICE_THREE, EYE,
    EYE_SLASH, FLOPPY_DISK_BACK, FOLDER_DASHED, INFO, LINE_SEGMENT, LINK_BREAK, LINK_SIMPLE,
    SUBTRACT_SQUARE, SWAP, TRASH, VECTOR_THREE, WARNING,
};
use nalgebra::Vector3;

use crate::{
    core::{
        App,
        config::ui::B_PER_MIB,
        history::ModelAction,
        project::{
            Collection, RenameState,
            model::{MeshUnit, MeshWarnings},
        },
    },
    interface::{
        components::{being_edited, dragger, grid, history_tracked_model},
        keybinds::Keybind,
        panels::models::{UiAction, shortcut},
    },
    task::{FileDialog, FlipWinding, MeshRepair, MeshSave, ReloadModel, SplitBodies},
};

const WARN_DEFECTIVE: &str = "This mesh is defective, it may produce unexpected results when sliced.\nConsider running it through a mesh repair tool.";
const WARN_OUT_OF_BOUNDS: &str = "This mesh extends beyond the printer volume and will be cut off.";
const SUPPORTABLE_TIP: &str =
    "Acceleration structures have been built for this mesh.\nSupports can be placed or generated.";
const UNIT_TIP: &str = "Common polygon mesh formats like .stl and .obj don't include unit information. For this reason you need to manually specify the units for each mesh, although Millimeters, the default, is often correct.";
const HOLLOW_TIP: &str = "Removes the inside of the model when slicing, keeping walls of the given thickness. Saves resin and reduces suction forces.\nAdd drain holes so uncured resin can escape the cavity. Cavities smaller than 0.05 mL are left solid.";
const DRAIN_HOLE_TIP: &str = "Drain holes let uncured resin flow out of the cavity and air in, which avoids suction forces while printing.\nSelect the drain hole tool (4) in the viewport toolbar, then click on the model to place one. Placing two holes near the bottom of the print works best.";
const CUSTOM_UNIT_TIP: &str = "To use a custom unit, input the conversion factor from the input unit to Millimeters. For example if your mesh was defined in feet (for some reason) you would use 30.48.";

pub fn model_entry(
    app: &mut App,
    ui: &mut Ui,
    model: usize,
    idx: usize,
    dragged: bool,
) -> Response {
    let model = &mut app.project.models[model];
    let id = model.id;

    let (rect, response) =
        ui.allocate_exact_size(vec2(ui.available_width(), 18.0), Sense::click_and_drag());

    let selected = app.state.selected.contains_model(id);
    let color = if selected && !dragged {
        ui.visuals().selection.bg_fill
    } else if response.hovered() || dragged {
        ui.visuals().code_bg_color
    } else if idx % 2 == 1 {
        ui.visuals().faint_bg_color
    } else {
        ui.style().noninteractive().bg_fill
    };

    let rect_margin = rect.expand2(vec2(2.0, ui.spacing().item_spacing.y / 2.0));
    ui.painter().rect_filled(rect_margin, 2.0, color);

    if response.clicked() {
        app.state
            .selected
            .model_clicked(id, ui.input(|x| x.modifiers.shift));
    }

    let indent = !dragged && model.collection.is_some();
    let ui_builder =
        UiBuilder::new().max_rect(rect.with_min_x(rect.min.x + [0.0, 12.0][indent as usize]));
    ui.scope_builder(ui_builder, |ui| {
        ui.horizontal(|ui| {
            ui.visuals_mut().override_text_color = selected.then_some(Color32::WHITE);
            ui.visuals_mut().button_frame = false;

            if ui
                .button(if model.hidden { EYE_SLASH } else { EYE })
                .on_hover_text(if model.hidden { "Show" } else { "Hide" })
                .clicked()
            {
                app.history
                    .track_model(model.id, ModelAction::Hidden(model.hidden));
                model.hidden ^= true;
            }

            if matches!(model.ui.rename, RenameState::None) {
                Label::new(&model.name).selectable(false).ui(ui);
            } else {
                let mut text_edit = TextEdit::singleline(&mut model.name).show(ui);
                if matches!(model.ui.rename, RenameState::Starting) {
                    text_edit.response.request_focus();
                    (text_edit.state.cursor)
                        .set_char_range(Some(CCursorRange::select_all(&text_edit.galley)));
                    text_edit.state.store(ui.ctx(), text_edit.response.id);
                    model.ui.rename = RenameState::Editing;
                }

                let editing = being_edited(&text_edit.response);
                (!editing).then(|| model.ui.rename = RenameState::None);

                history_tracked_model(
                    (editing, ui, &mut app.history),
                    (id, || ModelAction::Name(model.name.clone())),
                )
            }

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let icon = |icon: &str| Label::new(icon).selectable(false);

                if !model.warnings.is_empty() {
                    let count = model.warnings.bits().count_ones();
                    let mut warn = icon(&format!("{WARNING}{}", subscript_number(count))).ui(ui);
                    for warning in model.warnings.iter() {
                        let desc = match warning {
                            MeshWarnings::Defective => WARN_DEFECTIVE,
                            MeshWarnings::OutOfBounds => WARN_OUT_OF_BOUNDS,
                            _ => unreachable!(),
                        };
                        warn = warn.on_hover_text(desc);
                    }
                }

                if model.has_acceleration_structures() {
                    icon(LINE_SEGMENT).ui(ui).on_hover_text(SUPPORTABLE_TIP);
                }
            });
        });
    });

    response
}

pub fn model_properties(
    app: &mut App,
    ui: &mut Ui,
    ctx: &Context,
    action: &mut UiAction,
    i: usize,
) {
    let kb = &app.config.keybinds;
    let model = &mut app.project.models[i];

    let platform = &app.project.slice_config.platform_size;
    let id = model.id;

    ui.horizontal_wrapped(|ui| {
        if shortcut(
            kb,
            ui.button(concatcp!(CURSOR_TEXT, " Rename")),
            Keybind::Rename,
        ) {
            model.ui.rename = RenameState::Starting;
        }

        if shortcut(kb, ui.button(concatcp!(TRASH, " Delete")), Keybind::Delete) {
            *action = UiAction::Remove(i);
            app.state.selected.clear();
        }

        if shortcut(
            kb,
            ui.button(concatcp!(COPY, " Duplicate")),
            Keybind::Duplicate,
        ) {
            *action = UiAction::Duplicate(i);
        }

        if shortcut(
            kb,
            ui.button(concatcp!(FOLDER_DASHED, " Collect")),
            Keybind::Collect,
        ) {
            let collection = Collection::new_unnamed();
            model.collection = Some(collection.id);
            app.project.collections.push(collection);
        }

        if shortcut(
            kb,
            ui.button(concatcp!(ARROW_LINE_DOWN, " Align to Bed")),
            Keybind::Align,
        ) {
            let old_pos = model.mesh.position();
            app.history.track_model(id, ModelAction::Position(old_pos));

            model.align_to_bed();
            model.update_oob(platform);
        }

        if ui.button(concatcp!(SWAP, " Replace")).clicked() {
            let (id, name) = (model.id, model.name.clone());
            app.tasks.add(FileDialog::pick_file(
                ("Mesh", &["stl", "obj"]),
                move |_app, path, tasks| {
                    let task = ReloadModel::new(id, name, path.to_path_buf());
                    tasks.push(Box::new(task));
                },
            ));
        }

        if let Some(file) = &model.file
            && ui
                .button(concatcp!(ARROWS_COUNTER_CLOCKWISE, " Reload"))
                .clicked()
        {
            let task = ReloadModel::new(model.id, model.name.clone(), file.clone());
            app.tasks.add(task);
        }

        ui.menu_button(concatcp!(CUBE, " Mesh"), |ui| {
            if shortcut(
                kb,
                ui.button(concatcp!(SUBTRACT_SQUARE, " Split Bodies")),
                Keybind::Split,
            ) {
                app.tasks.add(SplitBodies::new(model));
            }

            if ui
                .button(concatcp!(VECTOR_THREE, " Flip Winding Order"))
                .clicked()
            {
                app.tasks.add(FlipWinding::new(model));
            }

            if ui.button(concatcp!(BANDAIDS, " Repair")).clicked() {
                app.tasks.add(MeshRepair::new(model));
            }

            ui.menu_button(concatcp!(FLOPPY_DISK_BACK, " Export"), |ui| {
                for format in mesh_format::Format::ALL {
                    if ui
                        .button(format!("{} ({})", format.name(), format.extension()))
                        .clicked()
                    {
                        let mesh = model.mesh.inner().clone();
                        app.tasks.add(FileDialog::save_file(
                            ("Mesh", &[format.extension()]),
                            move |_app, path, tasks| {
                                let path = path.with_extension(format.extension());
                                tasks.push(Box::new(MeshSave::new(path, format, mesh)));
                            },
                        ));
                    }
                }
            });
        });
    });

    CollapsingHeader::new("Transform")
        .default_open(true)
        .show(ui, |ui| {
            grid("model_props_grid").show(ui, |ui| {
                ui.label("Position");
                ui.horizontal(|ui| {
                    let mut position = model.mesh.position();
                    let editing = dragger::vec3(ui, position.as_mut(), |x| x);
                    history_tracked_model(
                        (editing, ui, &mut app.history),
                        (id, || ModelAction::Position(model.mesh.position())),
                    );
                    (editing && model.mesh.position() != position)
                        .then(|| model.set_position(platform, position));
                    ui.take_available_width();
                });
                ui.end_row();

                ui.label("Scale");

                ui.horizontal(|ui| {
                    let factor = model.unit.conversion();
                    let mut scale = model.mesh.scale() / factor;

                    let editing = if model.ui.locked_scale {
                        dragger::vec3_proportional(ui, scale.as_mut(), |x| {
                            x.speed(0.01).range(0.001..=f32::MAX)
                        })
                    } else {
                        dragger::vec3(ui, scale.as_mut(), |x| {
                            x.speed(0.01).range(0.001..=f32::MAX)
                        })
                    };
                    history_tracked_model(
                        (editing, ui, &mut app.history),
                        (id, || ModelAction::Scale(model.mesh.scale())),
                    );

                    scale *= factor;
                    (editing && model.mesh.scale() != scale)
                        .then(|| model.set_scale(platform, scale));

                    model.ui.locked_scale ^= ui
                        .button([LINK_BREAK, LINK_SIMPLE][model.ui.locked_scale as usize])
                        .clicked();
                });
                ui.end_row();

                ui.label("Rotation");
                let mut rotation = rad_to_deg(model.mesh.rotation());
                let editing = dragger::vec3(ui, rotation.as_mut(), |x| x.suffix("°"));
                history_tracked_model(
                    (editing, ui, &mut app.history),
                    (id, || ModelAction::Rotation(model.mesh.rotation())),
                );

                if editing {
                    let rad = deg_to_rad(rotation);
                    (model.mesh.rotation() != rad).then(|| model.set_rotation(platform, rad));
                }

                ui.end_row();

                ui.label("Color");
                ui.horizontal(|ui| {
                    let editing = Popup::is_id_open(ctx, ui.auto_id_with("popup"));
                    let original_color = model.color;
                    let mut color = model.color;
                    ui.color_edit_button_rgb(color.as_slice_mut())
                        .changed()
                        .then(|| model.color = color);
                    history_tracked_model(
                        (editing, ui, &mut app.history),
                        (id, || ModelAction::Color(original_color)),
                    );

                    if ui.button(concatcp!(DICE_THREE, " Random")).clicked() {
                        app.history.track_model(id, ModelAction::Color(model.color));
                        model.randomize_color();
                    }
                });
            });
        });

    ui.collapsing("Hollowing", |ui| {
        grid("model_hollow_grid").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Hollow");
                ui.label(INFO).on_hover_text(HOLLOW_TIP);
            });
            let mut enabled = model.hollow.enabled;
            if ui.checkbox(&mut enabled, "").changed() {
                app.history
                    .track_model(id, ModelAction::Hollow(model.hollow));
                model.hollow.enabled = enabled;
            }
            ui.end_row();

            ui.label("Wall Thickness");
            let mut wall = model.hollow.wall_thickness.get::<Milimeter>();
            let editing = being_edited(
                &ui.add_enabled(
                    model.hollow.enabled,
                    DragValue::new(&mut wall)
                        .range(0.5..=20.0)
                        .speed(0.05)
                        .max_decimals(2)
                        .suffix(" mm"),
                ),
            );

            history_tracked_model(
                (editing, ui, &mut app.history),
                (id, || ModelAction::Hollow(model.hollow)),
            );
            editing.then(|| model.hollow.wall_thickness = Milimeters::new(wall));
            ui.end_row();

            ui.horizontal(|ui| {
                ui.label("Drain Holes");
                ui.label(INFO).on_hover_text(DRAIN_HOLE_TIP);
            });
            ui.horizontal(|ui| {
                let count = model.drain_holes.len();
                if count == 0 && model.hollow.enabled {
                    let color = ui.visuals().warn_fg_color;
                    ui.colored_label(color, concatcp!(WARNING, " None"))
                        .on_hover_text(DRAIN_HOLE_TIP);
                } else {
                    ui.label(count.to_string());
                }

                if count > 0 && ui.button(concatcp!(TRASH, " Clear")).clicked() {
                    let old = ModelAction::DrainHoles(model.drain_holes.clone());
                    app.history.track_model(id, old);
                    model.drain_holes.clear();
                }
            });
            ui.end_row();

            ui.label("Hole Diameter");
            DragValue::new(&mut app.state.drain_hole.diameter)
                .range(0.5..=10.0)
                .speed(0.05)
                .max_decimals(2)
                .suffix(" mm")
                .ui(ui)
                .on_hover_text("Diameter of the next drain holes you place.");
            ui.end_row();

            let mut remove = None;
            for (i, hole) in model.drain_holes.iter().enumerate() {
                ui.label(format!("Hole {}", i + 1));
                ui.horizontal(|ui| {
                    ui.label(format!("Ø {:.2} mm", hole.radius * 2.0));
                    let button = ui
                        .small_button(TRASH)
                        .on_hover_text("Remove this drain hole");
                    button.clicked().then(|| remove = Some(i));
                });
                ui.end_row();
            }

            if let Some(i) = remove {
                let old = ModelAction::DrainHoles(model.drain_holes.clone());
                app.history.track_model(id, old);
                model.drain_holes.remove(i);
            }
        });
    });

    ui.collapsing("Miscellaneous", |ui| {
        grid("model_props_grid").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Mesh Units");
                ui.label(INFO).on_hover_text(UNIT_TIP);
            });
            ui.horizontal(|ui| {
                let last_factor = model.unit.conversion();
                ComboBox::from_id_salt("units")
                    .selected_text(model.unit.name())
                    .show_ui(ui, |ui| {
                        for unit in MeshUnit::ALL {
                            ui.selectable_value(&mut model.unit, unit, unit.name());
                        }
                    });

                if let MeshUnit::Custom(factor) = &mut model.unit {
                    DragValue::new(factor).speed(0.1).ui(ui);
                    ui.label(INFO).on_hover_text(CUSTOM_UNIT_TIP);
                }

                let factor = model.unit.conversion();
                if factor != last_factor {
                    let scale = model.mesh.scale() * factor / last_factor;
                    model.set_scale(&app.project.slice_config.platform_size, scale);
                }
            });

            ui.end_row();

            ui.label("Exposure");
            let mut value = model.exposure as f32 / 2.55;
            let editing = being_edited(
                &DragValue::new(&mut value)
                    .range(0.0..=100.0)
                    .max_decimals(0)
                    .suffix("%")
                    .ui(ui),
            );

            history_tracked_model(
                (editing, ui, &mut app.history),
                (id, || ModelAction::RelativeExposure(model.exposure)),
            );
            editing.then(|| model.exposure = (value * 2.55).round() as u8);

            ui.end_row();

            ui.label("Faces");
            ui.horizontal(|ui| {
                ui.label(format!(
                    "{} ({:.2} MiB)",
                    separate_thousands(model.mesh.face_count()),
                    model.mesh.memory_size() as f32 / B_PER_MIB as f32
                ));
                ui.take_available_width();
            });
            ui.end_row();

            if let Some(volume) = model.volume().map(|v| v.get::<Centimeter>()) {
                ui.label("Volume");
                ui.label(format!("{volume:.2} cm³"));
                ui.end_row();
            }
        });
    });
}

fn rad_to_deg(pos: Vector3<f32>) -> Vector3<f32> {
    pos.map(|x| x.to_degrees())
}

fn deg_to_rad(pos: Vector3<f32>) -> Vector3<f32> {
    pos.map(|x| x.to_radians())
}

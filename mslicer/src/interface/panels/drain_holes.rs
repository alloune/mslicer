//! Drain hole placement tool: clicking on a model adds a drain hole at that
//! point, perpendicular to the surface.

use slicer::post_process::hollowing::DrainHole;

use crate::{
    core::{App, history::ModelAction},
    task::BuildAccelerationStructures,
};

/// Updates the drain hole under the cursor and places it when `clicked`.
pub fn drain_hole_placement(app: &mut App, clicked: bool) {
    app.state.drain_hole.preview = None;
    if app.state.workspace.is_moving {
        return;
    }

    let Some((origin, direction)) = app.hovered_ray() else {
        return;
    };

    // Closest model surface along the cursor ray.
    let radius = app.state.drain_hole.diameter / 2.0;
    let mut closest = None;
    for model in app.project.models.iter().filter(|x| !x.hidden) {
        let Some(bvh) = model.bvh.as_ref() else {
            continue;
        };
        let Some(hit) = bvh.intersect_ray(&model.mesh, origin, direction) else {
            continue;
        };

        let distance = (model.mesh.transform(&hit.position) - origin).norm();
        if closest.is_some_and(|(best, _, _)| best <= distance) {
            continue;
        }

        let hole = DrainHole {
            position: hit.position,
            normal: model.mesh.normal(hit.face),
            radius,
        };
        closest = Some((distance, model.id, hole));
    }

    if let Some((_, id, hole)) = closest {
        app.state.drain_hole.preview = Some((id, hole));
        if clicked && let Some(model) = app.project.model(id) {
            let old = ModelAction::DrainHoles(model.drain_holes.clone());
            model.drain_holes.push(hole);
            app.history.track_model(id, old);
        }
    } else if clicked
        && let Some(hover) = app.state.hovered_geometry
        && let Some(model) = app.project.model(hover.model)
        && !model.has_acceleration_structures()
    {
        // Ray casting needs the model's acceleration structures.
        app.tasks.add(BuildAccelerationStructures::new(model));
    }
}

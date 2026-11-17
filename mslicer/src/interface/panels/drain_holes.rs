//! Drain hole tool: clicking on a model adds a drain hole at that point,
//! perpendicular to the surface. With Ctrl held, the hovered hole turns red
//! and clicking removes it.

use common::units::Milimeter;
use nalgebra::Vector3;
use slicer::post_process::hollowing::DrainHole;

use crate::{
    core::{App, history::ModelAction, project::model::ModelId},
    task::BuildAccelerationStructures,
};

/// How far behind the model surface a hole can be hit and still count as
/// hovered (mm). The visible part of a hole sticks out of the surface.
const SURFACE_TOLERANCE: f32 = 0.5;

/// Updates the drain hole under the cursor. When `clicked`, places a new hole
/// or, if `remove` (Ctrl held), removes the hovered one.
pub fn drain_hole_placement(app: &mut App, clicked: bool, remove: bool) {
    let tool = &mut app.state.drain_hole;
    (tool.preview, tool.hovered) = (None, None);
    if app.state.workspace.is_moving {
        return;
    }

    let Some((origin, direction)) = app.hovered_ray() else {
        return;
    };

    let surface = closest_surface(app, origin, direction);
    if remove {
        let max_distance = surface.map_or(f32::INFINITY, |x| x.0 + SURFACE_TOLERANCE);
        let hovered = closest_hole(app, origin, direction, max_distance);
        app.state.drain_hole.hovered = hovered;

        if clicked
            && let Some((id, index)) = hovered
            && let Some(model) = app.project.model(id)
        {
            let old = ModelAction::DrainHoles(model.drain_holes.clone());
            model.drain_holes.remove(index);
            app.history.track_model(id, old);
            app.state.drain_hole.hovered = None;
        }
        return;
    }

    if let Some((_, id, hole)) = surface {
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

/// Closest visible model surface along the ray, with the hole that would be
/// placed there.
fn closest_surface(
    app: &App,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
) -> Option<(f32, ModelId, DrainHole)> {
    let radius = app.state.drain_hole.diameter / 2.0;
    let mut closest: Option<(f32, ModelId, DrainHole)> = None;

    for model in app.project.models.iter().filter(|x| !x.hidden) {
        let Some(bvh) = model.bvh.as_ref() else {
            continue;
        };
        let Some(hit) = bvh.intersect_ray(&model.mesh, origin, direction) else {
            continue;
        };

        let distance = (model.mesh.transform(&hit.position) - origin).norm();
        if closest.is_none_or(|(best, ..)| distance < best) {
            let hole = DrainHole {
                position: hit.position,
                normal: model.mesh.normal(hit.face),
                radius,
            };
            closest = Some((distance, model.id, hole));
        }
    }

    closest
}

/// Hole (model, index) whose drawn cylinder the ray hits first, if closer than
/// `max_distance`.
fn closest_hole(
    app: &App,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    max_distance: f32,
) -> Option<(ModelId, usize)> {
    let mut closest: Option<(f32, (ModelId, usize))> = None;

    for model in app.project.models.iter().filter(|x| !x.hidden) {
        let wall = model.hollow.wall_thickness.get::<Milimeter>();
        for (i, hole) in model.drain_holes.iter().enumerate() {
            let (start, end) = hole.world_axis(&model.mesh, wall);
            if let Some(distance) = ray_cylinder(origin, direction, start, end, hole.radius)
                && distance <= max_distance
                && closest.is_none_or(|(best, _)| distance < best)
            {
                closest = Some((distance, (model.id, i)));
            }
        }
    }

    closest.map(|(_, hole)| hole)
}

/// Distance along a (normalized) ray to its first intersection with a closed
/// cylinder going from `start` to `end`.
fn ray_cylinder(
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    start: Vector3<f32>,
    end: Vector3<f32>,
    radius: f32,
) -> Option<f32> {
    let length = (end - start).norm();
    if length <= 0.0 {
        return None;
    }

    let axis = (end - start) / length;
    let w = origin - start;
    let (direction_axial, w_axial) = (direction.dot(&axis), w.dot(&axis));
    let (direction_radial, w_radial) = (direction - axis * direction_axial, w - axis * w_axial);
    let on_axis = |t: f32| (0.0..=length).contains(&(w_axial + t * direction_axial));

    let mut closest = f32::INFINITY;

    // Side: distance to the axis equals the radius.
    let a = direction_radial.norm_squared();
    if a > 1e-12 {
        let b = 2.0 * w_radial.dot(&direction_radial);
        let c = w_radial.norm_squared() - radius * radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant >= 0.0 {
            let root = discriminant.sqrt();
            for t in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
                if t >= 0.0 && on_axis(t) {
                    closest = closest.min(t);
                }
            }
        }
    }

    // Caps: the planes at both ends, within the radius.
    if direction_axial.abs() > 1e-9 {
        for cap in [0.0, length] {
            let t = (cap - w_axial) / direction_axial;
            let radial = (w_radial + direction_radial * t).norm_squared();
            if t >= 0.0 && radial <= radius * radius {
                closest = closest.min(t);
            }
        }
    }

    closest.is_finite().then_some(closest)
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::ray_cylinder;

    /// Vertical cylinder of radius 1 from z = 0 to z = 2, centered on x = y = 0.
    fn hit(origin: [f32; 3], direction: [f32; 3]) -> Option<f32> {
        let direction = Vector3::from(direction).normalize();
        let (start, end) = (Vector3::zeros(), Vector3::new(0.0, 0.0, 2.0));
        ray_cylinder(Vector3::from(origin), direction, start, end, 1.0)
    }

    #[test]
    fn ray_hits_cap() {
        let t = hit([0.5, 0.0, 10.0], [0.0, 0.0, -1.0]).unwrap();
        assert!((t - 8.0).abs() < 1e-4, "{t}");
    }

    #[test]
    fn ray_hits_side() {
        let t = hit([-5.0, 0.0, 1.0], [1.0, 0.0, 0.0]).unwrap();
        assert!((t - 4.0).abs() < 1e-4, "{t}");
    }

    #[test]
    fn ray_hits_side_at_an_angle() {
        // Enters the side at x = -1, z = 1.
        let t = hit([-3.0, 0.0, 3.0], [1.0, 0.0, -1.0]).unwrap();
        assert!((t - 8_f32.sqrt()).abs() < 1e-4, "{t}");
    }

    #[test]
    fn ray_misses() {
        assert_eq!(hit([-5.0, 2.0, 1.0], [1.0, 0.0, 0.0]), None);
        assert_eq!(hit([-5.0, 0.0, 3.0], [1.0, 0.0, 0.0]), None);
        assert_eq!(hit([5.0, 0.0, 1.0], [1.0, 0.0, 0.0]), None);
    }
}

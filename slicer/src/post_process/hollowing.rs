//! Hollowing: removes the inside of models while keeping a wall of constant
//! thickness, to save resin and reduce suction forces during printing.
//!
//! The work is done on the sliced layers rather than on the mesh, which avoids
//! fragile mesh boolean operations:
//!
//! 1. Each model to hollow is voxelized on a coarse grid (a few voxels per wall
//!    thickness) using the regular scanline rasterizer.
//! 2. An exact 3D Euclidean distance transform gives, for every voxel inside
//!    the model, its distance to the surface. Unlike a 2D per-layer erosion,
//!    this also keeps the requested thickness on top, bottom and sloped
//!    surfaces.
//! 3. For every printed layer, the distance field is interpolated at each
//!    pixel and the pixels deeper than the wall thickness are cleared, working
//!    directly on the run length encoded layer data.

use std::{sync::Arc, time::Instant};

use common::{
    container::Run,
    progress::Progress,
    serde::{Deserializer, Serializer},
    slice::{Layer, SliceConfig},
    units::{Milimeter, Milimeters},
};
use nalgebra::{Vector2, Vector3};
use rayon::prelude::*;
use tracing::info;

use crate::{
    geometry::{bounds::Bounds3D, mesh::Mesh, partitioning::Segments1D},
    slicer::raster::{Segment, layer},
};

/// Default number of voxels across the wall thickness.
const VOXELS_PER_WALL: f32 = 8.0;
/// Smallest voxel size allowed, in mm.
const MIN_VOXEL_SIZE: f32 = 0.05;
/// Upper bound on the number of voxels of a distance field, to cap memory use
/// (each voxel is an f32). The voxel size is increased to stay under it.
const MAX_VOXELS: f64 = 48e6;
/// Number of empty voxels added around the model bounds, so the distance
/// transform always sees the outside of the model.
const PADDING: usize = 2;
/// Number of buckets used to accelerate plane intersections when voxelizing.
const SEGMENT_LAYERS: usize = 100;
/// Value used for 'infinite' distance in the distance transform.
const FAR: f32 = 1e20;

/// Per model hollowing settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HollowConfig {
    pub enabled: bool,
    /// Minimum thickness of the remaining walls.
    pub wall_thickness: Milimeters,
}

impl Default for HollowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            wall_thickness: Milimeters::new(2.0),
        }
    }
}

impl HollowConfig {
    pub fn serialize<T: Serializer>(&self, ser: &mut T) {
        ser.write_bool(self.enabled);
        ser.write_f32_be(self.wall_thickness.get::<Milimeter>());
    }

    pub fn deserialize<T: Deserializer>(des: &mut T) -> Self {
        Self {
            enabled: des.read_bool(),
            wall_thickness: Milimeters::new(des.read_f32_be()),
        }
    }
}

/// A model to hollow.
pub struct HollowTarget {
    /// The model's mesh, already transformed into platform space (x and y in
    /// pixels, z in mm), exactly as given to the slicer.
    pub mesh: Mesh,
    /// Minimum thickness of the remaining walls.
    pub wall_thickness: Milimeters,
}

/// Hollows the given models in already sliced layers. Layers must still be in
/// their original order and count (run this before variable layer height).
pub fn hollow(
    config: &SliceConfig,
    layers: &mut [Layer],
    targets: &[HollowTarget],
    progress: Progress,
) {
    if targets.is_empty() {
        progress.set_finished();
        return;
    }

    progress.set_total((targets.len() * (layers.len() + 1)) as u64);
    let width = config.platform_resolution.x as u64;
    let slice_height = config.slice_height.get::<Milimeter>();

    for target in targets {
        let start = Instant::now();
        let wall = target.wall_thickness.get::<Milimeter>();
        let Some(field) = DistanceField::build(config, &target.mesh, wall) else {
            progress.add_complete(layers.len() as u64 + 1);
            continue;
        };
        progress.add_complete(1);
        info!(
            "Built hollowing distance field {}×{}×{} (voxel {:.3} mm) in {:?}",
            field.size.x,
            field.size.y,
            field.size.z,
            field.voxel,
            start.elapsed()
        );

        let start = Instant::now();
        let height = config.platform_resolution.y;
        layers.par_iter_mut().enumerate().for_each(|(i, layer)| {
            // Same sampling height as the rasterizer uses for this layer.
            let z = i as f32 * slice_height;
            let intervals = field.interior_intervals(z, wall, width, height);
            if !intervals.is_empty() {
                let runs = clear_intervals(&layer.data, &intervals);
                layer.area = runs.iter().filter(|x| x.value > 0).map(|x| x.length).sum();
                layer.data = Arc::new(runs);
            }
            progress.add_complete(1);
        });
        info!("Hollowed layers in {:?}", start.elapsed());
    }

    progress.set_finished();
}

/// Distance to the model surface sampled on a regular grid. Positive inside
/// the model, negative outside.
struct DistanceField {
    /// Platform space position of the corner of the grid (px, px, mm).
    origin: Vector3<f32>,
    /// Size of a voxel in platform space (px, px, mm).
    cell: Vector3<f32>,
    /// Size of a voxel in mm.
    voxel: f32,
    /// Number of voxels along each axis.
    size: Vector3<usize>,
    /// Distances in mm, stored with z varying fastest: `(y * nx + x) * nz + z`.
    data: Vec<f32>,
}

impl DistanceField {
    fn build(config: &SliceConfig, mesh: &Mesh, wall: f32) -> Option<Self> {
        let bounds = Bounds3D::from_mesh(mesh);
        let px_per_mm = (config.platform_resolution.cast::<f32>())
            .component_div(&config.platform_size.xy().map(|x| x.get::<Milimeter>()))
            .push(1.0);
        let extent_mm = (bounds.max - bounds.min).component_div(&px_per_mm);
        if extent_mm.iter().any(|x| !x.is_finite() || *x <= 0.0) {
            return None;
        }

        let voxels = |voxel: f32| {
            (extent_mm / voxel)
                .map(|x| x.ceil() as f64 + 2.0 * PADDING as f64)
                .product()
        };
        let mut voxel = (wall / VOXELS_PER_WALL).max(MIN_VOXEL_SIZE);
        while voxels(voxel) > MAX_VOXELS {
            voxel *= 1.1;
        }
        if voxel > wall / 2.0 {
            info!("Model is large: hollowing voxel size raised to {voxel:.3} mm");
        }

        let cell = px_per_mm * voxel;
        let size = (extent_mm / voxel).map(|x| x.ceil() as usize + 2 * PADDING);
        let origin = bounds.min - cell * PADDING as f32;

        let mut data = voxelize(mesh, &bounds, origin, cell, size);
        distance_transform(&mut data, size);

        // Squared distances in voxels to signed distance in mm. The surface is
        // assumed to lie halfway between the last solid and first empty voxel.
        data.par_iter_mut()
            .for_each(|d| *d = d.sqrt() * voxel - voxel / 2.0);

        Some(Self {
            origin,
            cell,
            voxel,
            size,
            data,
        })
    }

    fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[(y * self.size.x + x) * self.size.z + z]
    }

    /// Returns the sorted, non-overlapping ranges of pixel indices (in the
    /// layer's row major order) that are deeper than `wall` mm in the model,
    /// for the layer sampled at height `z` (mm, platform space).
    fn interior_intervals(&self, z: f32, wall: f32, width: u64, height: u32) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let (nx, ny, nz) = (self.size.x, self.size.y, self.size.z);

        // Grid coordinates are expressed so that voxel center `i` is at `i`.
        let gz = (z - self.origin.z) / self.cell.z - 0.5;
        if !(gz >= 0.0 && gz <= (nz - 1) as f32) {
            return out;
        }
        let k0 = gz.floor() as usize;
        let (k1, tz) = ((k0 + 1).min(nz - 1), gz - k0 as f32);

        // Interpolate the field at this height once for the whole layer.
        let plane = (0..ny * nx)
            .map(|i| {
                let (y, x) = (i / nx, i % nx);
                lerp(self.get(x, y, k0), self.get(x, y, k1), tz)
            })
            .collect::<Vec<_>>();
        if plane.iter().all(|&d| d <= wall) {
            return out;
        }

        let (ox, oy) = (self.origin.x, self.origin.y);
        let (cx, cy) = (self.cell.x, self.cell.y);
        // Pixel (real valued) whose center is at grid coordinate `g`.
        let pixel_x = |g: f32| (g + 0.5) * cx + ox - 0.5;
        let pixel_y = |g: f32| (g + 0.5) * cy + oy - 0.5;

        let y_start = pixel_y(0.0).ceil().max(0.0) as u32;
        let y_end = (pixel_y((ny - 1) as f32).floor() + 1.0).clamp(0.0, height as f32) as u32;

        let mut row = vec![0.0; nx];
        for y in y_start..y_end {
            let gy = ((y as f32 + 0.5 - oy) / cy - 0.5).clamp(0.0, (ny - 1) as f32);
            let j0 = gy.floor() as usize;
            let (j1, ty) = ((j0 + 1).min(ny - 1), gy - j0 as f32);

            let (a, b) = (&plane[j0 * nx..][..nx], &plane[j1 * nx..][..nx]);
            let mut deep = false;
            for x in 0..nx {
                row[x] = lerp(a[x], b[x], ty);
                deep |= row[x] > wall;
            }
            if !deep {
                continue;
            }

            let base = y as u64 * width;
            let clamp = |x: f32| x.clamp(0.0, width as f32) as u64;
            for x in 0..nx - 1 {
                let (da, db) = (row[x], row[x + 1]);
                if da <= wall && db <= wall {
                    continue;
                }

                // Pixels whose center lies between voxel centers x and x + 1.
                let (start, end) = (
                    clamp(pixel_x(x as f32).ceil()),
                    clamp(pixel_x(x as f32 + 1.0).ceil()),
                );
                let (start, end) = if da > wall && db > wall {
                    (start, end)
                } else {
                    // The field is linear between both centers: find where it
                    // crosses the wall thickness.
                    let crossing = pixel_x(x as f32 + (wall - da) / (db - da));
                    if da > wall {
                        (start, end.min(clamp(crossing.ceil())))
                    } else {
                        (start.max(clamp(crossing.floor() + 1.0)), end)
                    }
                };

                if start < end {
                    push_interval(&mut out, base + start, base + end);
                }
            }
        }

        out
    }
}

/// Rasterizes the mesh into a grid of `size` voxels starting at `origin` with
/// voxels of `cell` size (platform space). Returns `FAR` for voxels inside the
/// mesh and zero for voxels outside, with x varying fastest:
/// `(z * ny + y) * nx + x`.
fn voxelize(
    mesh: &Mesh,
    bounds: &Bounds3D,
    origin: Vector3<f32>,
    cell: Vector3<f32>,
    size: Vector3<usize>,
) -> Vec<f32> {
    let (nx, ny) = (size.x, size.y);
    let segments = Segments1D::from_mesh(mesh, bounds, SEGMENT_LAYERS);

    let mut data = vec![0.0; size.product()];
    data.par_chunks_mut(nx * ny)
        .enumerate()
        .for_each(|(z, plane)| {
            let height = origin.z + (z as f32 + 0.5) * cell.z;
            let intersections = segments.intersect_plane(mesh, height);
            let segments = intersections.into_iter().map(|(pos, entering)| Segment {
                endpoints: pos
                    .map(|p| Vector2::new((p.x - origin.x) / cell.x, (p.y - origin.y) / cell.y)),
                entering,
                priority: 255,
                exposure: 255,
            });

            let (runs, _errors) = layer(1, Vector2::new(nx as u32, ny as u32), segments);
            let mut pixel = 0;
            for run in runs {
                let end = (pixel + run.length as usize).min(plane.len());
                if run.value > 0 {
                    plane[pixel..end].fill(FAR);
                }
                pixel = end;
            }
        });

    data
}

/// Exact squared Euclidean distance transform (Felzenszwalb & Huttenlocher),
/// computed separably along each axis. Input values are zero for 'background'
/// voxels and [`FAR`] elsewhere. Input layout is `(z * ny + y) * nx + x`, the
/// output is transposed to `(y * nx + x) * nz + z`.
fn distance_transform(data: &mut Vec<f32>, size: Vector3<usize>) {
    let (nx, ny, nz) = (size.x, size.y, size.z);
    let scratch = || Scratch::new(nx.max(ny).max(nz));

    // Along x: rows are contiguous.
    data.par_chunks_mut(nx)
        .for_each_init(scratch, |s, line| edt_1d(line, s));

    // Along y: within each z plane.
    data.par_chunks_mut(nx * ny)
        .for_each_init(scratch, |s, plane| {
            let mut column = vec![0.0; ny];
            for x in 0..nx {
                (0..ny).for_each(|y| column[y] = plane[y * nx + x]);
                edt_1d(&mut column, s);
                (0..ny).for_each(|y| plane[y * nx + x] = column[y]);
            }
        });

    // Along z: transpose so z lines become contiguous.
    let source = &*data;
    let mut transposed = vec![0.0; source.len()];
    transposed
        .par_chunks_mut(nz)
        .enumerate()
        .for_each_init(scratch, |s, (i, line)| {
            let (y, x) = (i / nx, i % nx);
            (0..nz).for_each(|z| line[z] = source[(z * ny + y) * nx + x]);
            edt_1d(line, s);
        });

    *data = transposed;
}

struct Scratch {
    vertices: Vec<usize>,
    bounds: Vec<f64>,
    out: Vec<f32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            vertices: vec![0; n],
            bounds: vec![0.0; n + 1],
            out: vec![0.0; n],
        }
    }
}

/// One dimensional squared distance transform of a sampled function: computes
/// the lower envelope of the parabolas rooted at each sample.
fn edt_1d(f: &mut [f32], scratch: &mut Scratch) {
    let n = f.len();
    if n == 0 {
        return;
    }

    let Scratch {
        vertices: v,
        bounds: z,
        out: d,
    } = scratch;
    let parabola = |q: usize| f[q] as f64 + (q * q) as f64;

    let mut k = 0;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s;
        loop {
            let p = v[k];
            s = (parabola(q) - parabola(p)) / (2.0 * (q - p) as f64);
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }

    k = 0;
    for (q, out) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let dist = q.abs_diff(v[k]) as f32;
        *out = dist * dist + f[v[k]];
    }

    f.copy_from_slice(&d[..n]);
}

/// Sets every pixel inside the given (sorted, non-overlapping) intervals to
/// zero.
fn clear_intervals(runs: &[Run], intervals: &[(u64, u64)]) -> Vec<Run> {
    let mut out = Vec::with_capacity(runs.len() + intervals.len() * 2);
    let mut intervals = intervals.iter().copied().peekable();
    let mut pos = 0;

    for run in runs {
        let (mut cursor, end) = (pos, pos + run.length);
        pos = end;

        if run.value == 0 {
            push_run(&mut out, run.length, 0);
            continue;
        }

        while cursor < end {
            while intervals.next_if(|&(_, stop)| stop <= cursor).is_some() {}
            match intervals.peek() {
                Some(&(start, stop)) if start < end => {
                    if start > cursor {
                        push_run(&mut out, start - cursor, run.value);
                        cursor = start;
                    }
                    let stop = stop.min(end);
                    push_run(&mut out, stop - cursor, 0);
                    cursor = stop;
                }
                _ => {
                    push_run(&mut out, end - cursor, run.value);
                    cursor = end;
                }
            }
        }
    }

    out
}

fn push_run(out: &mut Vec<Run>, length: u64, value: u8) {
    if length == 0 {
        return;
    }
    match out.last_mut() {
        Some(last) if last.value == value => last.length += length,
        _ => out.push(Run::new(length, value)),
    }
}

fn push_interval(out: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    match out.last_mut() {
        Some(last) if start <= last.1 => last.1 = last.1.max(end),
        _ => out.push((start, end)),
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use common::{container::rle, slice::SliceConfig, units::Milimeters};
    use nalgebra::{Vector2, Vector3};

    use super::*;
    use crate::slicer::{Slicer, SlicerModel};

    #[test]
    fn edt_matches_brute_force() {
        let input = [0.0, FAR, FAR, FAR, 0.0, FAR, FAR, FAR, FAR, FAR];
        let mut f = input.to_vec();
        let mut scratch = Scratch::new(f.len());
        edt_1d(&mut f, &mut scratch);

        for (q, &d) in f.iter().enumerate() {
            let expected = (input.iter().enumerate())
                .filter(|(_, x)| **x == 0.0)
                .map(|(p, _)| (q.abs_diff(p) as f32).powi(2))
                .fold(f32::MAX, f32::min);
            assert_eq!(d, expected, "at {q}");
        }
    }

    #[test]
    fn clears_intervals_in_runs() {
        let runs = [
            Run::new(4, 0),
            Run::new(6, 200),
            Run::new(5, 0),
            Run::new(5, 100),
        ];
        let out = clear_intervals(&runs, &[(2, 6), (8, 17)]);
        let expected = [
            Run::new(6, 0),
            Run::new(2, 200),
            Run::new(9, 0),
            Run::new(3, 100),
        ];
        assert_eq!(out, expected);
    }

    const PX: f32 = 0.1; // mm per pixel
    const LAYER: f32 = 0.1;
    const SIZE: u32 = 400; // platform resolution, 40 mm
    const WALL: f32 = 2.0;

    /// Slices the mesh (already in platform space) on a 40 mm platform with
    /// 0.1 mm pixels and layers, then hollows it.
    fn slice_and_hollow(mesh: Mesh) -> (Vec<Layer>, Vec<Layer>) {
        let config = SliceConfig {
            platform_resolution: Vector2::new(SIZE, SIZE),
            platform_size: Vector3::repeat(SIZE as f32 * PX).map(Milimeters::new),
            slice_height: Milimeters::new(LAYER),
            ..Default::default()
        };

        let models = vec![SlicerModel {
            bounds: Bounds3D::from_mesh(&mesh),
            mesh: mesh.clone(),
            exposure: 255,
        }];
        let solid = Slicer::new(config.clone(), models).slice_raster();

        let mut layers = solid.clone();
        let targets = [HollowTarget {
            mesh,
            wall_thickness: Milimeters::new(WALL),
        }];
        hollow(&config, &mut layers, &targets, Progress::new());
        (solid, layers)
    }

    fn decode(layer: &Layer) -> Vec<u8> {
        rle::decode_vec(layer.data.iter())
    }

    /// A 20 mm cube must keep a 2 mm wall on every side, top and bottom.
    #[test]
    fn hollows_cube_with_constant_walls() {
        // Cube from 10 mm to 30 mm in x / y (px 100 to 300) and 0 to 20 mm in
        // z, shifted down half a layer like the slicing runner does.
        let (lo, hi) = (100.0, 300.0);
        let (bottom, top) = (-LAYER / 2.0, 20.0 - LAYER / 2.0);
        let vertices = vec![
            Vector3::new(lo, lo, bottom),
            Vector3::new(hi, lo, bottom),
            Vector3::new(hi, hi, bottom),
            Vector3::new(lo, hi, bottom),
            Vector3::new(lo, lo, top),
            Vector3::new(hi, lo, top),
            Vector3::new(hi, hi, top),
            Vector3::new(lo, hi, top),
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let (solid, layers) = slice_and_hollow(Mesh::new(vertices, faces));
        assert_eq!(layers.len(), 200);
        assert!(layers[100].area < solid[100].area / 2);

        // Side walls at mid height: measure the solid run from the left edge.
        let data = decode(&layers[100]);
        let row = &data[200 * 400..201 * 400];
        let wall_mm = row[100..].iter().take_while(|&&x| x > 0).count() as f32 * PX;
        assert!((wall_mm - WALL).abs() <= 0.3, "side wall is {wall_mm} mm");

        // Outside of the cube is untouched and the center is empty.
        assert_eq!(row[50], 0);
        assert_eq!(row[200], 0);

        // Bottom and top walls along the center column.
        let center = (0..200).map(|i| decode(&layers[i])[200 * 400 + 200]);
        let center = center.collect::<Vec<_>>();
        let bottom_mm = center.iter().take_while(|&&x| x > 0).count() as f32 * LAYER;
        let top_mm = center.iter().rev().take_while(|&&x| x > 0).count() as f32 * LAYER;
        assert!(
            (bottom_mm - WALL).abs() <= 0.3,
            "bottom wall is {bottom_mm} mm"
        );
        assert!((top_mm - WALL).abs() <= 0.3, "top wall is {top_mm} mm");
    }

    /// A 10 mm radius sphere must become a shell between radius 8 and 10 mm:
    /// checks the wall thickness on curved and sloped surfaces in 3D.
    #[test]
    fn hollows_sphere_into_shell() {
        const RADIUS: f32 = 10.0;
        const TOLERANCE: f32 = 0.3;
        let center = Vector3::new(20.0, 20.0, 12.0); // mm

        // UV sphere, in platform space (x / y in pixels, z in mm).
        let (rings, segments) = (64, 128);
        let mut vertices = vec![
            Vector3::new(0.0, 0.0, RADIUS),
            Vector3::new(0.0, 0.0, -RADIUS),
        ];
        for ring in 1..rings {
            let theta = std::f32::consts::PI * ring as f32 / rings as f32;
            for segment in 0..segments {
                let phi = std::f32::consts::TAU * segment as f32 / segments as f32;
                let dir = Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                vertices.push(dir * RADIUS);
            }
        }
        let vertex = |ring: u32, segment: u32| 2 + (ring - 1) * segments + segment % segments;
        let mut faces = Vec::new();
        for segment in 0..segments {
            faces.push([0, vertex(1, segment), vertex(1, segment + 1)]);
            faces.push([
                1,
                vertex(rings - 1, segment + 1),
                vertex(rings - 1, segment),
            ]);
            for ring in 1..rings - 1 {
                let (a, b) = (vertex(ring, segment), vertex(ring, segment + 1));
                let (c, d) = (vertex(ring + 1, segment), vertex(ring + 1, segment + 1));
                faces.push([a, c, d]);
                faces.push([a, d, b]);
            }
        }
        let vertices = vertices
            .into_iter()
            .map(|v| {
                let v = v + center;
                Vector3::new(v.x / PX, v.y / PX, v.z - LAYER / 2.0)
            })
            .collect();
        let (solid, layers) = slice_and_hollow(Mesh::new(vertices, faces));

        let mut checked = 0;
        for (i, (solid, hollow)) in solid.iter().zip(&layers).enumerate() {
            let (solid, hollow) = (decode(solid), decode(hollow));
            let z = i as f32 * LAYER;
            for y in (0..SIZE as usize).step_by(3) {
                for x in (0..SIZE as usize).step_by(3) {
                    let index = y * SIZE as usize + x;
                    let point = Vector3::new((x as f32 + 0.5) * PX, (y as f32 + 0.5) * PX, z);
                    let r = (point - center).norm();
                    let inner = RADIUS - WALL;

                    if solid[index] == 0 {
                        assert_eq!(hollow[index], 0, "pixel added at r = {r}");
                    } else if r < inner - TOLERANCE {
                        assert_eq!(hollow[index], 0, "not hollowed at r = {r}");
                        checked += 1;
                    } else if r > inner + TOLERANCE {
                        assert!(hollow[index] > 0, "wall removed at r = {r}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 100_000);
    }
}

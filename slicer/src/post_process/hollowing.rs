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
//!
//! Drain holes are cylinders cleared in the same pass, computed analytically
//! for each row of pixels. Finally the voxel grid is flood filled from the
//! outside to find cavities that no drain hole reaches, where uncured resin
//! would stay trapped.

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
/// How far a drain hole starts outside of the model surface (mm), so it opens
/// cleanly even on curved surfaces.
const DRAIN_HOLE_OUTSIDE: f32 = 1.0;
/// How far a drain hole goes past the wall thickness (mm), so it always
/// reaches the cavity.
const DRAIN_HOLE_EXTRA_DEPTH: f32 = 1.0;
/// Cavities smaller than this (mm³) are left solid: they would save almost no
/// resin but trap some, often in thin parts no drain hole can reach.
pub const MIN_CAVITY_VOLUME: f32 = 50.0;

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

/// A drain hole placed on a model. Stored in the model's own (untransformed)
/// mesh coordinates so it follows the model when it is moved, rotated or
/// scaled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrainHole {
    /// Point on the model surface, in mesh coordinates.
    pub position: Vector3<f32>,
    /// Outward surface normal at that point, in mesh coordinates.
    pub normal: Vector3<f32>,
    /// Radius of the hole in mm.
    pub radius: f32,
}

impl DrainHole {
    /// Returns the hole axis in world space for the given (transformed) mesh,
    /// from just outside the surface to just past the wall.
    pub fn world_axis(&self, mesh: &Mesh, wall: f32) -> (Vector3<f32>, Vector3<f32>) {
        let point = mesh.transform(&self.position);
        let normal = mesh.transform_normal(&self.normal).normalize();
        (
            point + normal * DRAIN_HOLE_OUTSIDE,
            point - normal * (wall + DRAIN_HOLE_EXTRA_DEPTH),
        )
    }

    pub fn serialize<T: Serializer>(&self, ser: &mut T) {
        for value in self.position.iter().chain(self.normal.iter()) {
            ser.write_f32_be(*value);
        }
        ser.write_f32_be(self.radius);
    }

    pub fn deserialize<T: Deserializer>(des: &mut T) -> Self {
        let mut vector = || Vector3::new(des.read_f32_be(), des.read_f32_be(), des.read_f32_be());
        let (position, normal) = (vector(), vector());
        Self {
            position,
            normal,
            radius: des.read_f32_be(),
        }
    }
}

/// A cylinder to clear from the layers, in platform millimeters: x and y are
/// pixel coordinates multiplied by the pixel size, z is the same as in the
/// platform space mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub start: Vector3<f32>,
    pub end: Vector3<f32>,
    pub radius: f32,
}

/// A model to hollow.
pub struct HollowTarget {
    /// The model's mesh, already transformed into platform space (x and y in
    /// pixels, z in mm), exactly as given to the slicer.
    pub mesh: Mesh,
    /// Minimum thickness of the remaining walls.
    pub wall_thickness: Milimeters,
    /// Drain holes to cut through the walls.
    pub drain_holes: Vec<Cylinder>,
}

/// Outcome of hollowing one model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HollowReport {
    /// Volume in mm³ of each cavity that no drain hole reaches. Uncured resin
    /// stays trapped in them and they act as suction cups while printing.
    pub closed_cavities: Vec<f32>,
}

/// Hollows the given models in already sliced layers and cuts their drain
/// holes. Layers must still be in their original order and count (run this
/// before variable layer height). Returns one report per target.
pub fn hollow(
    config: &SliceConfig,
    layers: &mut [Layer],
    targets: &[HollowTarget],
    progress: Progress,
) -> Vec<HollowReport> {
    if targets.is_empty() {
        progress.set_finished();
        return Vec::new();
    }

    progress.set_total((targets.len() * (layers.len() + 1)) as u64);
    let (width, height) = (config.platform_resolution.x, config.platform_resolution.y);
    let slice_height = config.slice_height.get::<Milimeter>();
    let mm_per_px = (config.platform_size.xy().map(|x| x.get::<Milimeter>()))
        .component_div(&config.platform_resolution.cast::<f32>());

    let mut reports = Vec::with_capacity(targets.len());
    for target in targets {
        let start = Instant::now();
        let wall = target.wall_thickness.get::<Milimeter>();
        let Some(field) = DistanceField::build(config, &target.mesh, wall) else {
            progress.add_complete(layers.len() as u64 + 1);
            reports.push(HollowReport::default());
            continue;
        };
        reports.push(field.report(wall, &target.drain_holes, mm_per_px));
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
        layers.par_iter_mut().enumerate().for_each(|(i, layer)| {
            // Same sampling height as the rasterizer uses for this layer.
            let z = i as f32 * slice_height;
            let mut intervals = field.interior_intervals(z, wall, width as u64, height);
            for hole in target.drain_holes.iter() {
                hole.intervals(z, mm_per_px, width, height, &mut intervals);
            }

            if !intervals.is_empty() {
                let intervals = merge_intervals(intervals);
                let runs = clear_intervals(&layer.data, &intervals);
                layer.area = runs.iter().filter(|x| x.value > 0).map(|x| x.length).sum();
                layer.data = Arc::new(runs);
            }
            progress.add_complete(1);
        });
        info!("Hollowed layers in {:?}", start.elapsed());
    }

    progress.set_finished();
    reports
}

impl Cylinder {
    /// Appends the ranges of pixel indices inside this cylinder for the layer
    /// sampled at height `z` (mm). Pixels are tested at their center.
    fn intervals(
        &self,
        z: f32,
        mm_per_px: Vector2<f32>,
        width: u32,
        height: u32,
        out: &mut Vec<(u64, u64)>,
    ) {
        let (a, r) = (self.start, self.radius);
        let length = (self.end - a).norm();
        if length <= 0.0 || r <= 0.0 {
            return;
        }
        let u = (self.end - a) / length;

        let (min, max) = (
            a.inf(&self.end).add_scalar(-r),
            a.sup(&self.end).add_scalar(r),
        );
        if z < min.z || z > max.z {
            return;
        }

        // Pixel `p` has its center at `(p + 0.5) * mm_per_px`.
        let first = |v: f32, size: f32| (v / size - 0.5).ceil().max(0.0);
        let last = |v: f32, size: f32, count: u32| (v / size - 0.5).floor().min(count as f32 - 1.0);
        let (y_start, y_end) = (first(min.y, mm_per_px.y), last(max.y, mm_per_px.y, height));

        let mut y = y_start;
        while y <= y_end {
            // Points of this row are `a + (s, dy, dz)` with `s` free. Being
            // inside the infinite cylinder is a quadratic inequality in `s`,
            // being between both caps is linear in `s`.
            let (dy, dz) = ((y + 0.5) * mm_per_px.y - a.y, z - a.z);
            let c = dy * u.y + dz * u.z; // axial position at s = 0
            let q = dy * dy + dz * dz;

            let (qa, qb, qc) = (1.0 - u.x * u.x, -2.0 * u.x * c, q - c * c - r * r);
            let mut range = if qa > 1e-6 {
                let discriminant = qb * qb - 4.0 * qa * qc;
                if discriminant < 0.0 {
                    None
                } else {
                    let root = discriminant.sqrt();
                    Some(((-qb - root) / (2.0 * qa), (-qb + root) / (2.0 * qa)))
                }
            } else {
                // Axis along x: the distance to the axis does not depend on s.
                (qc <= 0.0).then_some((f32::NEG_INFINITY, f32::INFINITY))
            };

            if u.x.abs() > 1e-6 {
                let (s0, s1) = ((0.0 - c) / u.x, (length - c) / u.x);
                range = range.map(|(lo, hi)| (lo.max(s0.min(s1)), hi.min(s0.max(s1))));
            } else if c < 0.0 || c > length {
                range = None;
            }

            if let Some((lo, hi)) = range
                && lo <= hi
            {
                let start = first(a.x + lo, mm_per_px.x);
                let end = last(a.x + hi, mm_per_px.x, width);
                if start <= end {
                    let base = y as u64 * width as u64;
                    out.push((base + start as u64, base + end as u64 + 1));
                }
            }

            y += 1.0;
        }
    }

    /// Tests a point given in platform millimeters, with some extra radius.
    fn contains(&self, point: Vector3<f32>, margin: f32) -> bool {
        let axis = self.end - self.start;
        let length = axis.norm();
        let w = point - self.start;
        let t = w.dot(&axis) / length;
        t >= 0.0 && t <= length && (w.norm_squared() - t * t) <= (self.radius + margin).powi(2)
    }
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

        let mut field = Self {
            origin,
            cell,
            voxel,
            size,
            data,
        };
        field.fill_small_cavities(wall);
        Some(field)
    }

    /// Keeps the cavities smaller than [`MIN_CAVITY_VOLUME`] solid, by
    /// clamping their depth to the wall thickness.
    fn fill_small_cavities(&mut self, wall: f32) {
        let min_voxels = (MIN_CAVITY_VOLUME / self.voxel.powi(3)).ceil() as usize;
        let mut visited = vec![false; self.data.len()];
        let mut small = Vec::new();

        for seed in 0..self.data.len() {
            if visited[seed] || self.data[seed] <= wall {
                continue;
            }

            visited[seed] = true;
            let mut members = vec![seed];
            let data = &self.data;
            let count = flood_fill(self.size, seed, |i| {
                let enter = !visited[i] && data[i] > wall;
                if enter {
                    visited[i] = true;
                    (members.len() < min_voxels).then(|| members.push(i));
                }
                enter
            });

            if count < min_voxels {
                small.extend(members);
            }
        }

        for i in small {
            self.data[i] = wall;
        }
    }

    fn get(&self, x: usize, y: usize, z: usize) -> f32 {
        self.data[(y * self.size.x + x) * self.size.z + z]
    }

    /// Finds the cavities left by hollowing that are not connected to the
    /// outside through any drain hole, by flood filling the empty voxels from
    /// the border of the grid.
    fn report(&self, wall: f32, holes: &[Cylinder], mm_per_px: Vector2<f32>) -> HollowReport {
        const SOLID: u8 = 0;
        const EMPTY: u8 = 1;
        const REACHED: u8 = 2;

        let (nx, ny, nz) = (self.size.x, self.size.y, self.size.z);
        let index = |x: usize, y: usize, z: usize| (y * nx + x) * nz + z;
        let mut state = (self.data.iter())
            .map(|&d| if d > 0.0 && d <= wall { SOLID } else { EMPTY })
            .collect::<Vec<_>>();

        // Open the voxels crossed by drain holes.
        let center = |x: usize, y: usize, z: usize| {
            let px = self.origin
                + (Vector3::new(x, y, z).cast::<f32>().add_scalar(0.5)).component_mul(&self.cell);
            Vector3::new(px.x * mm_per_px.x, px.y * mm_per_px.y, px.z)
        };
        let to_grid = |mm: Vector3<f32>| {
            let px = Vector3::new(mm.x / mm_per_px.x, mm.y / mm_per_px.y, mm.z);
            (px - self.origin).component_div(&self.cell)
        };
        for hole in holes {
            let r = Vector3::repeat(hole.radius + self.voxel);
            let lo = to_grid(hole.start.inf(&hole.end) - r).map(|x| x.floor().max(0.0) as usize);
            let hi = to_grid(hole.start.sup(&hole.end) + r);
            let hi = Vector3::new(
                (hi.x.ceil() as usize).min(nx),
                (hi.y.ceil() as usize).min(ny),
                (hi.z.ceil() as usize).min(nz),
            );
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    for z in lo.z..hi.z {
                        if hole.contains(center(x, y, z), self.voxel / 2.0) {
                            state[index(x, y, z)] = EMPTY;
                        }
                    }
                }
            }
        }

        let fill = |state: &mut Vec<u8>, seed: usize| {
            state[seed] = REACHED;
            flood_fill(self.size, seed, |i| {
                let enter = state[i] == EMPTY;
                enter.then(|| state[i] = REACHED);
                enter
            })
        };

        // The corner of the grid is always padding, so outside the model.
        fill(&mut state, 0);

        // Whatever cavity voxel was not reached is part of a closed cavity.
        let voxel_volume = self.voxel.powi(3);
        let mut closed_cavities = Vec::new();
        for i in 0..state.len() {
            if state[i] == EMPTY && self.data[i] > wall {
                closed_cavities.push(fill(&mut state, i) as f32 * voxel_volume);
            }
        }

        HollowReport { closed_cavities }
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

/// 6-connected flood fill over a grid stored as `(y * nx + x) * nz + z`,
/// starting from `seed` (already entered). `enter` is called on neighbors and
/// returns whether to visit them: it must mark them so they are not entered
/// twice. Returns the number of visited voxels.
fn flood_fill(size: Vector3<usize>, seed: usize, mut enter: impl FnMut(usize) -> bool) -> usize {
    let (nx, ny, nz) = (size.x, size.y, size.z);
    let (step_x, step_y) = (nz, nz * nx);

    let mut count = 0;
    let mut stack = vec![seed];
    while let Some(i) = stack.pop() {
        count += 1;
        let (z, x, y) = (i % nz, (i / nz) % nx, i / step_y);
        let neighbors = [
            (x > 0).then(|| i - step_x),
            (x + 1 < nx).then(|| i + step_x),
            (y > 0).then(|| i - step_y),
            (y + 1 < ny).then(|| i + step_y),
            (z > 0).then(|| i - 1),
            (z + 1 < nz).then(|| i + 1),
        ];
        for next in neighbors.into_iter().flatten() {
            if enter(next) {
                stack.push(next);
            }
        }
    }

    count
}

fn merge_intervals(mut intervals: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    intervals.sort_unstable_by_key(|x| x.0);
    let mut out = Vec::with_capacity(intervals.len());
    for (start, end) in intervals {
        push_interval(&mut out, start, end);
    }
    out
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
        let (solid, layers, _) = slice_and_hollow_with_holes(mesh, Vec::new());
        (solid, layers)
    }

    fn slice_and_hollow_with_holes(
        mesh: Mesh,
        drain_holes: Vec<Cylinder>,
    ) -> (Vec<Layer>, Vec<Layer>, Vec<HollowReport>) {
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
            drain_holes,
        }];
        let reports = hollow(&config, &mut layers, &targets, Progress::new());
        (solid, layers, reports)
    }

    fn decode(layer: &Layer) -> Vec<u8> {
        rle::decode_vec(layer.data.iter())
    }

    /// Cube from 10 mm to 30 mm in x / y (px 100 to 300) and 0 to 20 mm in z,
    /// shifted down half a layer like the slicing runner does.
    fn cube() -> Mesh {
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
        Mesh::new(vertices, faces)
    }

    /// A 20 mm cube must keep a 2 mm wall on every side, top and bottom.
    #[test]
    fn hollows_cube_with_constant_walls() {
        let (solid, layers) = slice_and_hollow(cube());
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

    /// Without drain holes the cube cavity is reported as closed, with about
    /// the expected 16 mm³ volume.
    #[test]
    fn reports_closed_cavity() {
        let (_, _, reports) = slice_and_hollow_with_holes(cube(), Vec::new());
        let cavities = &reports[0].closed_cavities;
        assert_eq!(cavities.len(), 1);
        let expected = (20.0 - 2.0 * WALL).powi(3);
        assert!(
            (cavities[0] - expected).abs() < expected * 0.1,
            "{cavities:?}"
        );
    }

    /// A vertical hole through the top wall opens the cavity.
    #[test]
    fn drain_hole_through_top() {
        let top = 20.0 - LAYER / 2.0;
        let hole = Cylinder {
            start: Vector3::new(20.0, 20.0, top + 1.0),
            end: Vector3::new(20.0, 20.0, top - WALL - 1.0),
            radius: 1.5,
        };
        let (solid, layers, reports) = slice_and_hollow_with_holes(cube(), vec![hole]);
        assert!(reports[0].closed_cavities.is_empty());

        // In the top wall: the hole is empty, the wall around it is not.
        let i = 190; // z = 19 mm
        let (before, data) = (decode(&solid[i]), decode(&layers[i]));
        let pixel = |x: f32, y: f32| (y / PX) as usize * SIZE as usize + (x / PX) as usize;
        assert!(before[pixel(20.0, 20.0)] > 0);
        assert_eq!(data[pixel(20.0, 20.0)], 0);
        assert_eq!(data[pixel(21.3, 20.0)], 0);
        assert!(data[pixel(21.8, 20.0)] > 0);
        assert!(data[pixel(20.0, 18.2)] > 0);

        // The hole area matches a 1.5 mm radius disk.
        let removed = (solid[i].area - layers[i].area) as f32 * PX * PX;
        let disk = std::f32::consts::PI * 1.5 * 1.5;
        assert!((removed - disk).abs() < disk * 0.1, "{removed} mm² removed");
    }

    /// A horizontal hole along x (axis parallel to the rows) through the
    /// left wall, at mid height.
    #[test]
    fn drain_hole_through_side() {
        let hole = Cylinder {
            start: Vector3::new(9.0, 20.0, 10.0),
            end: Vector3::new(10.0 + WALL + 1.0, 20.0, 10.0),
            radius: 1.0,
        };
        let (_, layers, reports) = slice_and_hollow_with_holes(cube(), vec![hole]);
        assert!(reports[0].closed_cavities.is_empty());

        let data = decode(&layers[100]);
        let row = |y: f32| &data[(y / PX) as usize * SIZE as usize..][..SIZE as usize];
        // Through the wall at the hole center, solid wall beside it.
        assert!(row(20.0)[100..121].iter().all(|&x| x == 0));
        assert!(row(21.5)[100..115].iter().all(|&x| x > 0));
        // Layers above the hole are untouched.
        let data = decode(&layers[115]);
        assert!(data[200 * SIZE as usize + 105] > 0);
    }

    /// Compares the analytic row intervals with a per pixel test, for random
    /// cylinders in every orientation and non square pixels.
    #[test]
    fn cylinder_intervals_match_point_test() {
        let (width, height) = (60, 50);
        let mm_per_px = Vector2::new(0.1, 0.13);
        let mut seed = 0x2545_f491_u32;
        let mut random = |lo: f32, hi: f32| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            lo + (hi - lo) * (seed as f32 / u32::MAX as f32)
        };

        for _ in 0..200 {
            let mut point = || Vector3::new(random(0.0, 6.0), random(0.0, 6.5), random(0.0, 3.0));
            let cylinder = Cylinder {
                start: point(),
                end: point(),
                radius: random(0.1, 1.5),
            };

            for z in [0.5, 1.0, 1.5, 2.0, 2.5] {
                let mut intervals = Vec::new();
                cylinder.intervals(z, mm_per_px, width, height, &mut intervals);
                let mut inside = vec![false; (width * height) as usize];
                for (start, end) in intervals {
                    inside[start as usize..end as usize].fill(true);
                }

                for (i, &inside) in inside.iter().enumerate() {
                    let (x, y) = ((i % width as usize) as f32, (i / width as usize) as f32);
                    let center = Vector3::new((x + 0.5) * mm_per_px.x, (y + 0.5) * mm_per_px.y, z);
                    if inside {
                        assert!(cylinder.contains(center, 1e-3), "{cylinder:?} {center:?}");
                    } else {
                        assert!(!cylinder.contains(center, -1e-3), "{cylinder:?} {center:?}");
                    }
                }
            }
        }
    }

    /// A 7 mm cube would only have a 27 mm³ cavity: it is left solid and
    /// nothing is reported.
    #[test]
    fn small_cavities_stay_solid() {
        let (lo, hi) = (165.0, 235.0); // 16.5 mm to 23.5 mm
        let (bottom, top) = (-LAYER / 2.0, 7.0 - LAYER / 2.0);
        let mut vertices = Vec::new();
        for z in [bottom, top] {
            for (x, y) in [(lo, lo), (hi, lo), (hi, hi), (lo, hi)] {
                vertices.push(Vector3::new(x, y, z));
            }
        }
        let faces = cube().faces().to_vec();
        let (solid, layers, reports) =
            slice_and_hollow_with_holes(Mesh::new(vertices, faces), Vec::new());

        assert!(reports[0].closed_cavities.is_empty());
        for (solid, layer) in solid.iter().zip(&layers) {
            assert_eq!(solid.area, layer.area);
        }
    }
}

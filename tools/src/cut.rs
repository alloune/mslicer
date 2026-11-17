//! Cuts a mesh in two with a plane, closing both cut faces, and optionally
//! adds tenons so the parts can be glued back perfectly aligned: pegs stick
//! out of the part below the plane and matching pockets (slightly larger, by
//! the clearance) are sunk into the part above.
//!
//! No mesh boolean operations are used: the triangles crossing the plane are
//! split with intersection points shared by both parts, the cut faces are
//! triangulated with circular holes where the tenons go, and the peg and
//! pocket walls are attached to those holes. Both parts come out closed.

use std::collections::HashMap;

use nalgebra::{Vector2, Vector3};
use slicer::geometry::mesh::orthogonal_basis;

/// Number of sides of the tenon cylinders.
const TENON_SIDES: usize = 32;
/// Chamfer on the tip of the pegs (mm), to ease insertion.
const TENON_CHAMFER: f32 = 0.4;
/// Extra depth of the pockets (mm), so pegs never bottom out.
const POCKET_EXTRA_DEPTH: f32 = 0.3;
/// Minimum material around a pocket and between two tenons (mm).
const TENON_MARGIN: f32 = 1.0;

/// A cutting plane, in the same space as the mesh vertices.
#[derive(Debug, Clone, Copy)]
pub struct CutPlane {
    pub point: Vector3<f32>,
    /// Points towards the part that receives the pockets.
    pub normal: Vector3<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TenonConfig {
    pub enabled: bool,
    /// Maximum number of tenons per separate cut face.
    pub count: u8,
    /// Diameter of the pegs (mm).
    pub diameter: f32,
    /// Length of the pegs (mm).
    pub height: f32,
    /// Gap between a peg and its pocket, on the radius (mm).
    pub clearance: f32,
}

impl Default for TenonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            count: 2,
            diameter: 5.0,
            height: 5.0,
            clearance: 0.15,
        }
    }
}

/// A closed triangle mesh.
#[derive(Debug, Default, Clone)]
pub struct CutPart {
    pub vertices: Vec<Vector3<f32>>,
    pub faces: Vec<[u32; 3]>,
}

#[derive(Debug, Default)]
pub struct CutResult {
    /// Part on the opposite side of the plane normal, with the pegs.
    pub below: CutPart,
    /// Part on the side of the plane normal, with the pockets.
    pub above: CutPart,
    /// Tenons placed.
    pub tenons: usize,
    /// Separate cut faces (connected areas of the cut).
    pub cut_faces: usize,
    /// Separate cut faces too small (or too thin above) for any tenon.
    pub faces_without_tenons: usize,
    /// Cut outlines that could not be closed (holes in the mesh). Their cut
    /// face is left open.
    pub open_outlines: usize,
}

/// Cuts the mesh. Returns `None` if the plane does not cross it.
pub fn cut(
    vertices: &[Vector3<f32>],
    faces: &[[u32; 3]],
    plane: CutPlane,
    tenons: &TenonConfig,
) -> Option<CutResult> {
    let normal = plane.normal.normalize();
    let mut point = plane.point;

    // Keep vertices off the plane, which makes every crossing well defined.
    let extent = vertices.iter().map(|v| v.abs().max()).fold(1.0, f32::max);
    let epsilon = extent * 1e-6;
    let distances = loop {
        let distances = vertices
            .iter()
            .map(|v| (v - point).dot(&normal))
            .collect::<Vec<_>>();
        if distances.iter().all(|d| d.abs() > epsilon) {
            break distances;
        }
        point += normal * epsilon * 3.0;
    };

    if distances.iter().all(|&d| d > 0.0) || distances.iter().all(|&d| d < 0.0) {
        return None;
    }

    let mut split = Split::new(vertices, &distances);
    for &face in faces {
        split.face(face);
    }

    let [u, v] = orthogonal_basis(normal);
    let project = |p: Vector3<f32>| Vector2::new((p - point).dot(&u), (p - point).dot(&v));

    let (loops, open_outlines) = split.loops();
    let mut regions = regions(&loops, &split, &project);

    let mut result = CutResult {
        open_outlines,
        cut_faces: regions.len(),
        ..Default::default()
    };
    let to_3d = |p: Vector2<f32>| point + u * p.x + v * p.y;

    for region in regions.iter_mut() {
        if tenons.enabled && tenons.count > 0 {
            let centers = place_tenons(region, tenons, vertices, faces, normal, &to_3d);
            result.tenons += centers.len();
            result.faces_without_tenons += centers.is_empty() as usize;
            region.tenons = centers;
        }
        split.cap(region, tenons, normal, &to_3d);
    }

    result.below = split.below;
    result.above = split.above;
    Some(result)
}

/// The two parts being built while splitting the faces.
struct Split<'a> {
    vertices: &'a [Vector3<f32>],
    distances: &'a [f32],

    below: CutPart,
    above: CutPart,
    below_map: HashMap<u32, u32>,
    above_map: HashMap<u32, u32>,

    /// Crossing point of each cut edge: (position, index in below, index in above).
    crossings: HashMap<(u32, u32), (Vector3<f32>, u32, u32)>,
    /// Directed edges of the cut face of the part below (normal along the
    /// plane normal), between crossing edge keys.
    outline: Vec<((u32, u32), (u32, u32))>,
}

impl<'a> Split<'a> {
    fn new(vertices: &'a [Vector3<f32>], distances: &'a [f32]) -> Self {
        Self {
            vertices,
            distances,
            below: CutPart::default(),
            above: CutPart::default(),
            below_map: HashMap::new(),
            above_map: HashMap::new(),
            crossings: HashMap::new(),
            outline: Vec::new(),
        }
    }

    fn vertex(&mut self, index: u32, above: bool) -> u32 {
        let (part, map) = match above {
            true => (&mut self.above, &mut self.above_map),
            false => (&mut self.below, &mut self.below_map),
        };
        *map.entry(index).or_insert_with(|| {
            part.vertices.push(self.vertices[index as usize]);
            part.vertices.len() as u32 - 1
        })
    }

    /// Point where edge (a, b) crosses the plane, shared by both parts.
    fn crossing(&mut self, a: u32, b: u32) -> (u32, u32) {
        let key = (a.min(b), a.max(b));
        if let Some(&(_, below, above)) = self.crossings.get(&key) {
            return (below, above);
        }

        let (p, q) = (key.0 as usize, key.1 as usize);
        let (dp, dq) = (self.distances[p], self.distances[q]);
        let position = self.vertices[p] + (self.vertices[q] - self.vertices[p]) * (dp / (dp - dq));

        self.below.vertices.push(position);
        self.above.vertices.push(position);
        let below = self.below.vertices.len() as u32 - 1;
        let above = self.above.vertices.len() as u32 - 1;
        self.crossings.insert(key, (position, below, above));
        (below, above)
    }

    fn face(&mut self, face: [u32; 3]) {
        let side = face.map(|i| self.distances[i as usize] > 0.0);
        if side[0] == side[1] && side[1] == side[2] {
            let face = face.map(|i| self.vertex(i, side[0]));
            let part = if side[0] {
                &mut self.above
            } else {
                &mut self.below
            };
            part.faces.push(face);
            return;
        }

        // Rotate (keeping the winding) so the vertex alone on its side is first.
        let lone = (0..3).find(|&i| side[i] != side[(i + 1) % 3] && side[i] != side[(i + 2) % 3]);
        let lone = lone.unwrap();
        let [l, m1, m2] = [face[lone], face[(lone + 1) % 3], face[(lone + 2) % 3]];
        let lone_above = side[lone];

        let i1 = self.crossing(l, m1);
        let i2 = self.crossing(l, m2);
        let pick =
            |(below, above): (u32, u32), is_above: bool| if is_above { above } else { below };

        let l = self.vertex(l, lone_above);
        let (m1, m2) = (self.vertex(m1, !lone_above), self.vertex(m2, !lone_above));
        let (lone_part, other_part) = match lone_above {
            true => (&mut self.above, &mut self.below),
            false => (&mut self.below, &mut self.above),
        };
        lone_part
            .faces
            .push([l, pick(i1, lone_above), pick(i2, lone_above)]);
        let (j1, j2) = (pick(i1, !lone_above), pick(i2, !lone_above));
        other_part.faces.push([j1, m1, m2]);
        other_part.faces.push([j1, m2, j2]);

        // The cut face must hold the reverse of the edge along the plane:
        // i2 -> i1 on the lone side, i1 -> i2 on the other.
        let key = |a: u32, b: u32| (a.min(b), a.max(b));
        let (k1, k2) = (
            key(face[lone], face[(lone + 1) % 3]),
            key(face[lone], face[(lone + 2) % 3]),
        );
        let below_edge = if lone_above { (k1, k2) } else { (k2, k1) };
        self.outline.push(below_edge);
    }

    /// Chains the outline edges into closed loops (of crossing keys). Returns
    /// the loops and the number of chains that could not be closed.
    fn loops(&self) -> (Vec<Vec<(u32, u32)>>, usize) {
        let mut next = HashMap::<(u32, u32), Vec<(u32, u32)>>::new();
        for &(a, b) in self.outline.iter() {
            next.entry(a).or_default().push(b);
        }

        let mut loops = Vec::new();
        let mut open = 0;
        while let Some(&start) = next.keys().next() {
            let mut chain = vec![start];
            let mut current = start;
            let closed = loop {
                let Some(targets) = next.get_mut(&current) else {
                    break false;
                };
                let target = targets.pop().unwrap();
                if targets.is_empty() {
                    next.remove(&current);
                }
                if target == start {
                    break true;
                }
                chain.push(target);
                current = target;
            };

            if closed && chain.len() >= 3 {
                loops.push(chain);
            } else {
                open += 1;
            }
        }

        (loops, open)
    }

    /// Closes the cut faces of one region, with the tenon holes, pegs and
    /// pockets.
    fn cap(
        &mut self,
        region: &Region,
        tenons: &TenonConfig,
        normal: Vector3<f32>,
        to_3d: &impl Fn(Vector2<f32>) -> Vector3<f32>,
    ) {
        let peg_radius = tenons.diameter / 2.0;
        let pocket_radius = peg_radius + tenons.clearance;

        // Polygon for the triangulation: outer ring, holes, tenon circles.
        let mut points = Vec::<Vector2<f32>>::new();
        let mut below_index = Vec::<u32>::new();
        let mut above_index = Vec::<u32>::new();
        let mut hole_starts = Vec::new();
        let mut peg_rings = Vec::new();
        let mut pocket_rings = Vec::new();

        for (i, (ring, polygon)) in region.rings.iter().zip(&region.polygons).enumerate() {
            (i > 0).then(|| hole_starts.push(points.len()));
            for (key, &point) in ring.iter().zip(polygon) {
                let (_, below, above) = self.crossings[key];
                points.push(point);
                below_index.push(below);
                above_index.push(above);
            }
        }

        for &center in region.tenons.iter() {
            hole_starts.push(points.len());
            let (mut pegs, mut pockets) = (Vec::new(), Vec::new());
            for i in 0..TENON_SIDES {
                let angle = std::f32::consts::TAU * i as f32 / TENON_SIDES as f32;
                let direction = Vector2::new(angle.cos(), angle.sin());
                // The triangulation only needs one shape per hole: both rings
                // have the same angles, so the pocket ring (larger) is used
                // for the above part and the peg ring for the below part.
                points.push(center + direction * pocket_radius);
                self.below
                    .vertices
                    .push(to_3d(center + direction * peg_radius));
                self.above
                    .vertices
                    .push(to_3d(center + direction * pocket_radius));
                pegs.push(self.below.vertices.len() as u32 - 1);
                pockets.push(self.above.vertices.len() as u32 - 1);
            }
            below_index.extend(pegs.iter());
            above_index.extend(pockets.iter());
            peg_rings.push((center, pegs));
            pocket_rings.push((center, pockets));
        }

        let flat = points
            .iter()
            .flat_map(|p| [p.x as f64, p.y as f64])
            .collect::<Vec<_>>();
        let Ok(triangles) = earcutr::earcut(&flat, &hole_starts, 2) else {
            return;
        };

        // Counter clockwise in 2D, so the below cap faces along the normal.
        let mut triangles = (triangles.chunks_exact(3))
            .map(|t| {
                let area = (points[t[1]] - points[t[0]]).perp(&(points[t[2]] - points[t[0]]));
                if area >= 0.0 {
                    [t[0], t[1], t[2]]
                } else {
                    [t[0], t[2], t[1]]
                }
            })
            .collect::<Vec<_>>();
        restore_dropped_points(&points, &mut triangles);

        for [a, b, c] in triangles {
            self.below
                .faces
                .push([below_index[a], below_index[b], below_index[c]]);
            self.above
                .faces
                .push([above_index[a], above_index[c], above_index[b]]);
        }

        let height = tenons.height;
        let chamfer = TENON_CHAMFER.min(height / 2.0).min(peg_radius / 2.0);
        for (center, ring) in peg_rings {
            let ring_at = |part: &mut CutPart, radius: f32, offset: f32| {
                let start = part.vertices.len() as u32;
                for i in 0..TENON_SIDES {
                    let angle = std::f32::consts::TAU * i as f32 / TENON_SIDES as f32;
                    let direction = Vector2::new(angle.cos(), angle.sin());
                    part.vertices
                        .push(to_3d(center + direction * radius) + normal * offset);
                }
                (start..start + TENON_SIDES as u32).collect::<Vec<_>>()
            };

            let shoulder = ring_at(&mut self.below, peg_radius, height - chamfer);
            let tip = ring_at(&mut self.below, peg_radius - chamfer, height);
            wall(&mut self.below, &ring, &shoulder, false);
            wall(&mut self.below, &shoulder, &tip, false);

            self.below.vertices.push(to_3d(center) + normal * height);
            let top = self.below.vertices.len() as u32 - 1;
            fan(&mut self.below, top, &tip, false);
        }

        let depth = height + POCKET_EXTRA_DEPTH;
        for (center, ring) in pocket_rings {
            let start = self.above.vertices.len() as u32;
            for &i in ring.iter() {
                let position = self.above.vertices[i as usize] + normal * depth;
                self.above.vertices.push(position);
            }
            let bottom = (start..start + TENON_SIDES as u32).collect::<Vec<_>>();
            wall(&mut self.above, &ring, &bottom, true);

            self.above.vertices.push(to_3d(center) + normal * depth);
            let center = self.above.vertices.len() as u32 - 1;
            fan(&mut self.above, center, &bottom, true);
        }
    }
}

/// Earcut skips points lying on a straight edge of the outline. The faces
/// around the cut still use them, so they are put back by splitting the cap
/// triangle along that edge, keeping both parts closed.
fn restore_dropped_points(points: &[Vector2<f32>], triangles: &mut Vec<[usize; 3]>) {
    let mut used = vec![false; points.len()];
    triangles.iter().flatten().for_each(|&i| used[i] = true);

    for point in (0..points.len()).filter(|&i| !used[i]) {
        let p = points[point];
        let on_edge = |a: usize, b: usize| {
            let (a, b) = (points[a], points[b]);
            let ab = b - a;
            let length = ab.norm_squared();
            let t = (p - a).dot(&ab);
            let off = ab.perp(&(p - a)).abs() / length.sqrt().max(1e-12);
            t > 0.0 && t < length && off < 1e-4 * length.sqrt().max(1.0)
        };

        let found = triangles.iter().enumerate().find_map(|(i, &[a, b, c])| {
            [(a, b, c), (b, c, a), (c, a, b)]
                .into_iter()
                .find(|&(x, y, _)| on_edge(x, y))
                .map(|edge| (i, edge))
        });

        if let Some((i, (a, b, c))) = found {
            triangles[i] = [a, point, c];
            triangles.push([point, b, c]);
        }
    }
}

/// Faces between two rings (counter clockwise around the plane normal) along
/// the normal, facing away from the axis, or towards it if `inward`.
fn wall(part: &mut CutPart, low: &[u32], high: &[u32], inward: bool) {
    for i in 0..low.len() {
        let j = (i + 1) % low.len();
        if inward {
            part.faces.push([low[j], low[i], high[i]]);
            part.faces.push([low[j], high[i], high[j]]);
        } else {
            part.faces.push([low[i], low[j], high[j]]);
            part.faces.push([low[i], high[j], high[i]]);
        }
    }
}

/// Disk closing a ring, facing along the normal, or against it if `flip`.
fn fan(part: &mut CutPart, center: u32, ring: &[u32], flip: bool) {
    for i in 0..ring.len() {
        let j = (i + 1) % ring.len();
        part.faces.push(match flip {
            false => [center, ring[i], ring[j]],
            true => [center, ring[j], ring[i]],
        });
    }
}

/// One connected cut face: an outer ring and its holes, in crossing keys.
struct Region {
    rings: Vec<Vec<(u32, u32)>>,
    /// 2D coordinates of the rings (in the plane), in the same order.
    polygons: Vec<Vec<Vector2<f32>>>,
    /// Centers of the tenons placed on this face.
    tenons: Vec<Vector2<f32>>,
}

impl Region {
    fn contains(&self, p: Vector2<f32>) -> bool {
        self.polygons.iter().filter(|ring| inside(ring, p)).count() % 2 == 1
    }

    fn distance_to_edges(&self, p: Vector2<f32>) -> f32 {
        (self.polygons.iter())
            .flat_map(|ring| (0..ring.len()).map(move |i| (ring[i], ring[(i + 1) % ring.len()])))
            .map(|(a, b)| segment_distance(p, a, b))
            .fold(f32::INFINITY, f32::min)
    }
}

/// Groups the loops into outer rings (counter clockwise around the normal) and
/// the holes they contain.
fn regions(
    loops: &[Vec<(u32, u32)>],
    split: &Split,
    project: &impl Fn(Vector3<f32>) -> Vector2<f32>,
) -> Vec<Region> {
    let polygons = (loops.iter())
        .map(|ring| {
            ring.iter()
                .map(|key| project(split.crossings[key].0))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let areas = polygons.iter().map(|x| signed_area(x)).collect::<Vec<_>>();

    // An inside out mesh gives clockwise outer rings.
    let sign = if areas.iter().sum::<f32>() >= 0.0 {
        1.0
    } else {
        -1.0
    };

    let outers = (0..loops.len())
        .filter(|&i| areas[i] * sign > 0.0)
        .collect::<Vec<_>>();
    let mut rings = outers.iter().map(|&i| vec![i]).collect::<Vec<_>>();
    for hole in (0..loops.len()).filter(|&i| areas[i] * sign <= 0.0) {
        // Smallest outer ring containing the hole.
        let owner = (0..outers.len())
            .filter(|&o| inside(&polygons[outers[o]], polygons[hole][0]))
            .min_by(|&a, &b| areas[outers[a]].abs().total_cmp(&areas[outers[b]].abs()));
        if let Some(owner) = owner {
            rings[owner].push(hole);
        }
    }

    rings
        .into_iter()
        .map(|indices| Region {
            rings: indices.iter().map(|&i| loops[i].clone()).collect(),
            polygons: indices.iter().map(|&i| polygons[i].clone()).collect(),
            tenons: Vec::new(),
        })
        .collect()
}

/// Spreads up to `count` tenons over the region, where the pocket and its
/// margin fit inside the cut face and there is enough material above it.
fn place_tenons(
    region: &Region,
    tenons: &TenonConfig,
    vertices: &[Vector3<f32>],
    faces: &[[u32; 3]],
    normal: Vector3<f32>,
    to_3d: &impl Fn(Vector2<f32>) -> Vector3<f32>,
) -> Vec<Vector2<f32>> {
    let pocket_radius = tenons.diameter / 2.0 + tenons.clearance;
    let clearance = pocket_radius + TENON_MARGIN;
    let depth = tenons.height + POCKET_EXTRA_DEPTH + TENON_MARGIN;

    let (min, max) = (region.polygons[0].iter()).fold(
        (
            Vector2::repeat(f32::INFINITY),
            Vector2::repeat(f32::NEG_INFINITY),
        ),
        |(min, max), p| (min.inf(p), max.sup(p)),
    );
    let size = max - min;
    let step = (pocket_radius / 2.0).max(size.max() / 100.0);

    // Candidate centers with their distance to the edges of the cut face.
    let mut candidates = Vec::new();
    let mut y = min.y + step / 2.0;
    while y < max.y {
        let mut x = min.x + step / 2.0;
        while x < max.x {
            let p = Vector2::new(x, y);
            if region.contains(p) {
                let distance = region.distance_to_edges(p);
                (distance >= clearance).then(|| candidates.push((p, distance)));
            }
            x += step;
        }
        y += step;
    }

    // Is there enough material above the cut for the pocket?
    let deep_enough = |center: Vector2<f32>| {
        let rim = (0..4).map(|i| {
            let angle = std::f32::consts::FRAC_PI_2 * i as f32;
            center + Vector2::new(angle.cos(), angle.sin()) * pocket_radius
        });
        std::iter::once(center).chain(rim).all(|p| {
            let origin = to_3d(p);
            ray_distance(vertices, faces, origin, normal).is_some_and(|t| t >= depth)
        })
    };

    // Farthest point sampling: start from the widest spot, then keep adding
    // the candidate farthest from the tenons already placed.
    let mut placed = Vec::<Vector2<f32>>::new();
    let separation = 2.0 * pocket_radius + 2.0 * TENON_MARGIN;
    while placed.len() < tenons.count as usize {
        let score = |p: Vector2<f32>, distance: f32| match placed.is_empty() {
            true => distance,
            false => placed
                .iter()
                .map(|t| (t - p).norm())
                .fold(f32::INFINITY, f32::min),
        };

        let mut best = None::<(f32, usize)>;
        for (i, &(p, distance)) in candidates.iter().enumerate() {
            if placed.iter().any(|t| (t - p).norm() < separation) {
                continue;
            }
            let score = score(p, distance);
            if best.is_none_or(|(s, _)| score > s) {
                best = Some((score, i));
            }
        }

        let Some((_, i)) = best else {
            break;
        };
        let (p, _) = candidates.swap_remove(i);
        if deep_enough(p) {
            placed.push(p);
        }
    }

    placed
}

/// Distance along the ray to the first triangle it crosses (Möller–Trumbore).
fn ray_distance(
    vertices: &[Vector3<f32>],
    faces: &[[u32; 3]],
    origin: Vector3<f32>,
    direction: Vector3<f32>,
) -> Option<f32> {
    let mut closest = None::<f32>;
    for face in faces {
        let [a, b, c] = face.map(|i| vertices[i as usize]);
        let (e1, e2) = (b - a, c - a);
        let p = direction.cross(&e2);
        let det = e1.dot(&p);
        if det.abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / det;
        let s = origin - a;
        let u = s.dot(&p) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(&e1);
        let v = direction.dot(&q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = e2.dot(&q) * inv;
        if t > 1e-6 && closest.is_none_or(|c| t < c) {
            closest = Some(t);
        }
    }
    closest
}

fn signed_area(polygon: &[Vector2<f32>]) -> f32 {
    (0..polygon.len())
        .map(|i| polygon[i].perp(&polygon[(i + 1) % polygon.len()]))
        .sum::<f32>()
        / 2.0
}

fn inside(polygon: &[Vector2<f32>], p: Vector2<f32>) -> bool {
    let mut inside = false;
    for i in 0..polygon.len() {
        let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
        if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
            inside = !inside;
        }
    }
    inside
}

fn segment_distance(p: Vector2<f32>, a: Vector2<f32>, b: Vector2<f32>) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(&ab) / ab.norm_squared().max(1e-12)).clamp(0.0, 1.0);
    (a + ab * t - p).norm()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nalgebra::Vector3;

    use super::*;

    /// Axis aligned box, faces pointing out (or in, for a cavity).
    fn cuboid(min: [f32; 3], max: [f32; 3], inward: bool, part: &mut CutPart) {
        let start = part.vertices.len() as u32;
        for z in [min[2], max[2]] {
            for (x, y) in [
                (min[0], min[1]),
                (max[0], min[1]),
                (max[0], max[1]),
                (min[0], max[1]),
            ] {
                part.vertices.push(Vector3::new(x, y, z));
            }
        }
        let faces = [
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
        for [a, b, c] in faces {
            let face = if inward { [a, c, b] } else { [a, b, c] };
            part.faces.push(face.map(|i| i + start));
        }
    }

    fn cube(min: [f32; 3], max: [f32; 3]) -> CutPart {
        let mut part = CutPart::default();
        cuboid(min, max, false, &mut part);
        part
    }

    fn cut_z(part: &CutPart, z: f32, tenons: &TenonConfig) -> CutResult {
        let plane = CutPlane {
            point: Vector3::new(0.0, 0.0, z),
            normal: Vector3::z(),
        };
        cut(&part.vertices, &part.faces, plane, tenons).unwrap()
    }

    /// Every directed edge must be matched by exactly one reverse edge.
    fn assert_closed(part: &CutPart) {
        let mut edges = HashMap::<(u32, u32), i32>::new();
        for &[a, b, c] in part.faces.iter() {
            for (p, q) in [(a, b), (b, c), (c, a)] {
                *edges.entry((p, q)).or_default() += 1;
            }
        }
        for (&(p, q), &count) in edges.iter() {
            assert_eq!(count, 1, "edge {p} -> {q} used {count} times");
            assert_eq!(edges.get(&(q, p)), Some(&1), "edge {p} -> {q} has no twin");
        }
    }

    fn volume(part: &CutPart) -> f32 {
        let v = |i: u32| part.vertices[i as usize];
        (part.faces.iter())
            .map(|&[a, b, c]| v(a).dot(&v(b).cross(&v(c))))
            .sum::<f32>()
            / 6.0
    }

    /// Area of the regular polygon used for tenons, of the given radius.
    fn ring_area(radius: f32) -> f32 {
        let n = TENON_SIDES as f32;
        n / 2.0 * radius * radius * (std::f32::consts::TAU / n).sin()
    }

    fn assert_close(actual: f32, expected: f32) {
        let error = (actual - expected).abs() / expected;
        assert!(error < 1e-3, "{actual} instead of {expected}");
    }

    #[test]
    fn plain_cut() {
        let tenons = TenonConfig {
            enabled: false,
            ..Default::default()
        };
        let result = cut_z(&cube([0.0; 3], [20.0; 3]), 7.0, &tenons);

        assert_closed(&result.below);
        assert_closed(&result.above);
        assert_close(volume(&result.below), 20.0 * 20.0 * 7.0);
        assert_close(volume(&result.above), 20.0 * 20.0 * 13.0);
        assert_eq!((result.tenons, result.open_outlines), (0, 0));
    }

    #[test]
    fn plane_outside_the_mesh() {
        let mesh = cube([0.0; 3], [20.0; 3]);
        let plane = CutPlane {
            point: Vector3::new(0.0, 0.0, 25.0),
            normal: Vector3::z(),
        };
        assert!(cut(&mesh.vertices, &mesh.faces, plane, &TenonConfig::default()).is_none());
    }

    #[test]
    fn cut_with_tenons() {
        let tenons = TenonConfig::default();
        let result = cut_z(&cube([0.0; 3], [20.0; 3]), 7.0, &tenons);
        assert_eq!(result.tenons, 2);
        assert_closed(&result.below);
        assert_closed(&result.above);

        // Pegs: prism up to the chamfer, then a frustum.
        let r = tenons.diameter / 2.0;
        let (a1, a2) = (ring_area(r), ring_area(r - TENON_CHAMFER));
        let frustum = TENON_CHAMFER / 3.0 * (a1 + a2 + (a1 * a2).sqrt());
        let peg = a1 * (tenons.height - TENON_CHAMFER) + frustum;
        assert_close(volume(&result.below), 20.0 * 20.0 * 7.0 + 2.0 * peg);

        let pocket = ring_area(r + tenons.clearance) * (tenons.height + POCKET_EXTRA_DEPTH);
        assert_close(volume(&result.above), 20.0 * 20.0 * 13.0 - 2.0 * pocket);

        // Pegs stick out of the cut face, pockets go into the upper part.
        let top = |part: &CutPart| part.vertices.iter().map(|v| v.z).fold(f32::MIN, f32::max);
        assert!((top(&result.below) - (7.0 + tenons.height)).abs() < 1e-4);
        let deepest = (result.above.vertices.iter())
            .filter(|v| v.x > 0.5 && v.x < 19.5 && v.y > 0.5 && v.y < 19.5 && v.z > 7.0)
            .map(|v| v.z)
            .fold(f32::MIN, f32::max);
        assert!((deepest - (7.0 + tenons.height + POCKET_EXTRA_DEPTH)).abs() < 1e-4);
    }

    #[test]
    fn tenons_are_spread_apart() {
        let tenons = TenonConfig {
            count: 3,
            ..Default::default()
        };
        let result = cut_z(&cube([0.0; 3], [40.0, 20.0, 20.0]), 10.0, &tenons);
        assert_eq!(result.tenons, 3);

        // The peg tips are at the top of the lower part.
        let tips = (result.below.vertices.iter())
            .filter(|v| (v.z - 15.0).abs() < 1e-4)
            .map(|v| v.xy())
            .collect::<Vec<_>>();
        let spread = tips.iter().map(|p| p.x).fold(f32::MIN, f32::max)
            - tips.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        assert!(spread > 20.0, "tenons only {spread} mm apart");
        assert_closed(&result.below);
        assert_closed(&result.above);
    }

    #[test]
    fn no_tenon_without_room() {
        let tenons = TenonConfig::default();

        // Cut face only 4 mm wide.
        let result = cut_z(&cube([0.0; 3], [20.0, 4.0, 20.0]), 10.0, &tenons);
        assert_eq!((result.tenons, result.faces_without_tenons), (0, 1));
        assert_closed(&result.below);
        assert_closed(&result.above);

        // Only 3 mm of material above the cut, not enough for a pocket.
        let result = cut_z(&cube([0.0; 3], [20.0; 3]), 17.0, &tenons);
        assert_eq!((result.tenons, result.faces_without_tenons), (0, 1));
    }

    /// Cube with a closed cubic cavity: the cut face is a square ring, and the
    /// tenons must avoid the hole in it.
    #[test]
    fn cut_face_with_hole() {
        let mut mesh = cube([0.0; 3], [30.0; 3]);
        cuboid([10.0; 3], [20.0; 3], true, &mut mesh);

        let tenons = TenonConfig::default();
        let result = cut_z(&mesh, 15.0, &tenons);
        assert_closed(&result.below);
        assert_closed(&result.above);
        assert_eq!(result.open_outlines, 0);
        assert!(result.tenons >= 1);

        let r = tenons.diameter / 2.0;
        let (a1, a2) = (ring_area(r), ring_area(r - TENON_CHAMFER));
        let peg = a1 * (tenons.height - TENON_CHAMFER)
            + TENON_CHAMFER / 3.0 * (a1 + a2 + (a1 * a2).sqrt());
        let expected = 30.0 * 30.0 * 15.0 - 10.0 * 10.0 * 5.0 + result.tenons as f32 * peg;
        assert_close(volume(&result.below), expected);

        // No peg inside the cavity.
        for v in result.below.vertices.iter().filter(|v| v.z > 15.0 + 1e-3) {
            let in_hole = (10.0..20.0).contains(&v.x) && (10.0..20.0).contains(&v.y);
            assert!(!in_hole, "peg in the cavity at {v:?}");
        }
    }

    #[test]
    fn tilted_plane() {
        let mesh = cube([0.0; 3], [20.0; 3]);
        let plane = CutPlane {
            point: Vector3::new(10.0, 10.0, 10.0),
            normal: Vector3::new(1.0, 0.5, 1.0),
        };
        let result = cut(&mesh.vertices, &mesh.faces, plane, &TenonConfig::default()).unwrap();
        assert_closed(&result.below);
        assert_closed(&result.above);
        assert!(result.tenons >= 1);

        // The plane goes through the center: both halves are mirror images,
        // apart from the tenons.
        let r = 2.5_f32;
        let pegs = volume(&result.below) - 4000.0;
        let pockets = 4000.0 - volume(&result.above);
        assert!(pegs > 0.0 && pockets > 0.0);
        assert!(pegs < result.tenons as f32 * ring_area(r) * 5.0 * 1.01);
    }
}

#[cfg(test)]
mod slicing {
    use common::{container::rle, slice::SliceConfig, units::Milimeters};
    use nalgebra::{Vector2, Vector3};
    use slicer::{
        geometry::{bounds::Bounds3D, mesh::Mesh},
        slicer::{Slicer, SlicerModel},
    };

    use super::*;

    /// The cut parts slice without defects, and a layer through the pegs has
    /// the expected area.
    #[test]
    fn cut_parts_slice_cleanly() {
        let mut cube = CutPart::default();
        for z in [0.0, 20.0] {
            for (x, y) in [(10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)] {
                cube.vertices.push(Vector3::new(x, y, z));
            }
        }
        cube.faces = vec![
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

        let tenons = TenonConfig::default();
        let plane = CutPlane {
            point: Vector3::new(0.0, 0.0, 10.0),
            normal: Vector3::z(),
        };
        let result = cut(&cube.vertices, &cube.faces, plane, &tenons).unwrap();
        assert_eq!(result.tenons, 2);

        // 0.05 mm pixels and layers on a 40 mm platform.
        let config = SliceConfig {
            platform_resolution: Vector2::new(800, 800),
            platform_size: Vector3::repeat(40.0).map(Milimeters::new),
            slice_height: Milimeters::new(0.05),
            ..Default::default()
        };
        let slice = |part: &CutPart| {
            let vertices = (part.vertices.iter())
                .map(|v| Vector3::new(v.x / 0.05, v.y / 0.05, v.z))
                .collect();
            let mesh = Mesh::new(vertices, part.faces.clone());
            let model = SlicerModel {
                bounds: Bounds3D::from_mesh(&mesh),
                mesh,
                exposure: 255,
            };
            Slicer::new(config.clone(), vec![model]).slice_raster()
        };

        let below = slice(&result.below);
        assert_eq!(below.iter().map(|x| x.defects).sum::<u64>(), 0);
        let above = slice(&result.above);
        assert_eq!(above.iter().map(|x| x.defects).sum::<u64>(), 0);

        // 2 mm above the cut: two peg sections in the lower part, the cube
        // minus two pockets in the upper part.
        let pixel_area = 0.05 * 0.05;
        let peg = std::f32::consts::PI * 2.5 * 2.5;
        let pocket = std::f32::consts::PI * 2.65 * 2.65;
        let area = |layers: &[common::slice::Layer], z: f32| {
            let data = rle::decode_vec(layers[(z / 0.05) as usize].data.iter());
            data.iter().filter(|&&x| x > 0).count() as f32 * pixel_area
        };
        let pegs = area(&below, 12.0);
        assert!(
            (pegs - 2.0 * peg).abs() < 2.0 * peg * 0.03,
            "{pegs} mm² of pegs"
        );
        let ring = area(&above, 12.0);
        let expected = 400.0 - 2.0 * pocket;
        assert!(
            (ring - expected).abs() < expected * 0.01,
            "{ring} mm² around the pockets"
        );
    }
}

#[cfg(test)]
mod teapot {
    use std::{collections::HashMap, fs::File, io::BufReader};

    use common::{progress::Progress, serde::ReaderDeserializer};
    use nalgebra::Vector3;

    use super::*;

    fn open_edges(faces: &[[u32; 3]]) -> usize {
        let mut edges = HashMap::<(u32, u32), i32>::new();
        for &[a, b, c] in faces {
            for (p, q) in [(a, b), (b, c), (c, a)] {
                *edges.entry((p, q)).or_default() += 1;
            }
        }
        (edges.keys())
            .filter(|&&(p, q)| edges.get(&(q, p)) != Some(&1))
            .count()
    }

    #[test]
    fn cut_teapot() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../mslicer/assets/teapot.stl");
        let des = ReaderDeserializer::new(BufReader::new(File::open(path).unwrap()));
        let mesh = mesh_format::load_mesh(des, mesh_format::Format::Stl, &Progress::new()).unwrap();
        let vertices = mesh.verts.iter().map(|v| v * 5.0).collect::<Vec<_>>();
        let original_open = open_edges(&mesh.faces);

        let plane = CutPlane {
            point: Vector3::new(0.0, 0.0, 20.0),
            normal: Vector3::z(),
        };
        let result = cut(&vertices, &mesh.faces, plane, &TenonConfig::default()).unwrap();
        eprintln!(
            "teapot: original open edges {original_open}, tenons {}, without {}, open outlines {}, below open {}, above open {}",
            result.tenons,
            result.faces_without_tenons,
            result.open_outlines,
            open_edges(&result.below.faces),
            open_edges(&result.above.faces),
        );
        if original_open == 0 {
            assert_eq!(result.open_outlines, 0);
            assert_eq!(open_edges(&result.below.faces), 0);
            assert_eq!(open_edges(&result.above.faces), 0);
        }
        assert!(result.tenons >= 1);
    }
}

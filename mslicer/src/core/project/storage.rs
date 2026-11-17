use std::{collections::HashMap, sync::Arc};

use anyhow::{Result, ensure};
use const_format::concatcp;
use nalgebra::Vector3;

use crate::core::project::{
    Collection, CollectionId, PostProcessing, Project, RenameState,
    model::{MeshUnit, Model},
};
use common::{
    progress::Progress,
    serde::{Deserializer, SerdeExt, Serializer},
    slice::SliceConfig,
};
use slicer::{
    geometry::mesh::{Mesh, MeshInner},
    post_process::{
        elephant_foot_fixer::ElephantFootFixer,
        hollowing::{DrainHole, HollowConfig},
        variable_layer_height::VariableLayerHeight,
    },
};

/// Project format version. Value should be incremented whenever the save format
/// changes, even in development so anyone using the dev/prerelease versions
/// don't ruin their projects.
///
/// - v18 (fork) — Added per model drain holes.
/// - v17 (fork) — Added per model hollowing settings.
/// - v16 (v0.10.0) — Added project notes and an ignored section with info about
///   the format in ASCII.
/// - v15 (v0.10.0) — Store first layers and transition layers as either a layer
///   count or a united distance.
/// - v14 (v0.10.0) — Separate XY and Z supersample antialiasing parameters. You
///   might not always want 3DAA or the (honestly slight) performance impact.
/// - v13 (v0.10.0) — Merge exposure config lift and retract distances. Since
///   these values really should always be the same.
/// - v12 (v0.9.0) — Store mesh units.
/// - v11 (v0.9.0) — Customizable variable layer height similarity threshold.
/// - v10 (v0.9.0) — Added variable layer height post processor.
/// - v9 (v0.8.0) — Added collections.
/// - v8 (v0.8.0) — Added exposure remapping curve to slice config.
/// - v7 (v0.7.0) — Added exposure delay option to exposure config.
/// - v6 (v0.7.0) — Store SliceMode (raster / vector) instead of the specific
///   Format to save to. Since slicing and encoding are now separated.
/// - v5 (v0.7.0) — Replaced the (inaccurate and incredibly slow) blur based
///   anti aliasing with a super sampling approach.
/// - v4 (v0.6.0) — Added relative exposure value to models.
/// - v3 (v0.6.0) — Added the PWM value to SliceConfig -> ExposureConfig.
/// - v2 (v0.5.0) — A complete rewrite using a custom serilizer/deserilizer
///   because of the bincode drama...
const VERSION: u16 = 18;
const DESCRIPTION: &str = concatcp!(
    "Saved by mslicer v",
    crate::VERSION,
    " (https://mslicer.com)"
);

struct ModelInfo {
    mesh: u32,
    collection: Option<CollectionId>,

    name: String,
    unit: MeshUnit,
    color: Vector3<f32>,
    exposure: u8,
    hidden: bool,

    position: Vector3<f32>,
    scale: Vector3<f32>,
    rotation: Vector3<f32>,

    hollow: HollowConfig,
    drain_holes: Vec<DrainHole>,
}

impl ModelInfo {
    pub fn new(mesh: u32, model: &Model) -> Self {
        Self {
            mesh,
            collection: model.collection,
            name: model.name.to_owned(),
            unit: model.unit,
            color: model.color.into(),
            exposure: model.exposure,
            hidden: model.hidden,
            position: model.mesh.position(),
            scale: model.mesh.scale(),
            rotation: model.mesh.rotation(),
            hollow: model.hollow,
            drain_holes: model.drain_holes.clone(),
        }
    }

    pub fn into_model(self, inner: Arc<MeshInner>) -> Model {
        let mut mesh = Mesh::from_inner(inner);
        mesh.set_position_unchecked(self.position);
        mesh.set_scale_unchecked(self.scale);
        mesh.set_rotation_unchecked(self.rotation);
        mesh.update_transformation_matrix();

        Model::from_mesh(mesh)
            .with_name(self.name)
            .with_color(self.color.into())
            .with_exposure(self.exposure)
            .with_hollow(self.hollow)
            .with_drain_holes(self.drain_holes)
            .with_hidden(self.hidden)
            .with_collection(self.collection)
            .with_unit(self.unit)
    }

    pub fn serialize<T: Serializer>(&self, ser: &mut T) {
        // Mesh reference
        ser.write_u32_be(self.mesh);

        // Model properties
        if let Some(collection) = self.collection {
            ser.write_bool(true);
            ser.write_u32_be(collection.raw());
        } else {
            ser.write_bool(false);
        }

        ser.write_u32_be(self.name.len() as u32);
        ser.write_bytes(self.name.as_bytes());
        self.unit.serialize(ser);
        Vector3::from(self.color).serialize(ser);
        ser.write_u8(self.exposure);
        ser.write_bool(self.hidden);

        // Mesh properties
        self.position.serialize(ser);
        self.scale.serialize(ser);
        self.rotation.serialize(ser);

        // Post processing
        self.hollow.serialize(ser);
        ser.write_u32_be(self.drain_holes.len() as u32);
        for hole in self.drain_holes.iter() {
            hole.serialize(ser);
        }
    }

    pub fn deserialize<T: Deserializer>(des: &mut T, version: u16) -> Self {
        Self {
            mesh: des.read_u32_be(),
            collection: if version < 9 {
                None
            } else {
                des.read_bool()
                    .then(|| CollectionId::from_raw(des.read_u32_be()))
            },
            name: {
                let name_len = des.read_u32_be();
                let data = des.read_bytes(name_len as usize);
                String::from_utf8_lossy(&data).into_owned()
            },
            unit: if version < 12 {
                MeshUnit::Millimeters
            } else {
                MeshUnit::deserialize(des)
            },
            color: Vector3::<f32>::deserialize(des),
            exposure: if version < 4 { 255 } else { des.read_u8() },
            hidden: des.read_bool(),
            position: Vector3::<f32>::deserialize(des),
            scale: Vector3::<f32>::deserialize(des),
            rotation: Vector3::<f32>::deserialize(des),
            hollow: if version < 17 {
                HollowConfig::default()
            } else {
                HollowConfig::deserialize(des)
            },
            drain_holes: if version < 18 {
                Vec::new()
            } else {
                let count = des.read_u32_be();
                (0..count).map(|_| DrainHole::deserialize(des)).collect()
            },
        }
    }
}

impl Project {
    pub fn serialize<T: Serializer>(&self, ser: &mut T, progress: Progress) {
        ser.write_u16_be(VERSION);

        ser.write_u32_be(DESCRIPTION.len() as u32);
        ser.write_bytes(DESCRIPTION.as_bytes());

        ser.write_u32_be(self.notes.len() as u32);
        ser.write_bytes(self.notes.as_bytes());

        self.slice_config.serialize(ser);
        self.post_processing.serialize(ser);

        let mut total = 0;
        let mut map = HashMap::new();
        let mut meshes = Vec::new();

        ser.write_u32_be(self.models.len() as u32);
        for model in self.models.iter() {
            let id = model.mesh.mesh_id();
            let mesh = match map.get(&id) {
                Some(mesh) => *mesh,
                None => {
                    let mesh = map.len() as u32;
                    meshes.push(model.mesh.inner().clone());
                    total += model.mesh.vertex_count() + model.mesh.face_count();
                    map.insert(id, mesh);
                    mesh
                }
            };

            let info = ModelInfo::new(mesh, model);
            info.serialize(ser);
        }

        progress.set_total(total as u64);
        ser.write_u32_be(meshes.len() as u32);
        (meshes.iter()).for_each(|mesh| serialize_mesh_inner(ser, mesh, &progress));

        ser.write_u32_be(self.collections.len() as u32);
        self.collections.iter().for_each(|c| c.serialize(ser));
        progress.set_finished();
    }

    pub fn deserialize<T: Deserializer>(des: &mut T, progress: Progress) -> Result<Self> {
        let version = des.read_u16_be();
        ensure!(
            (2..=VERSION).contains(&version),
            "Save version not supported."
        );

        let notes = if version >= 16 {
            let description_len = des.read_u32_be() as usize;
            des.advance_by(description_len);

            let notes_len = des.read_u32_be() as usize;
            String::from_utf8_lossy(&des.read_bytes(notes_len)).into_owned()
        } else {
            String::new()
        };

        let slice_config = SliceConfig::deserialize(des, version)?;
        let post_processing = PostProcessing::deserialize(des, version);

        let models = des.read_u32_be();
        let models = (0..models)
            .map(|_| ModelInfo::deserialize(des, version))
            .collect::<Vec<_>>();

        let meshes = des.read_u32_be();
        progress.set_total((des.size() - des.pos()) as u64);
        let meshes = (0..meshes)
            .map(|_| Arc::new(deserialize_mesh_inner(des, &progress)))
            .collect::<Vec<_>>();

        let models = (models.into_iter())
            .map(|x| {
                let mesh = meshes[x.mesh as usize].clone();
                x.into_model(mesh)
            })
            .collect();

        let collections = (0..des.read_u32_be())
            .map(|_| Collection::deserialize(des))
            .collect::<Vec<_>>();

        // kinda jank but it's whatever
        if let Some(max_id) = collections.iter().map(|x| x.id.raw()).max() {
            CollectionId::bump_past(max_id);
        }

        progress.set_finished();
        Ok(Self {
            path: None,
            notes,
            slice_config,
            support_config: Default::default(),
            post_processing,
            models,
            collections,
        })
    }
}

impl Collection {
    pub fn serialize<T: Serializer>(&self, ser: &mut T) {
        ser.write_u32_be(self.id.raw());
        ser.write_u32_be(self.name.len() as u32);
        ser.write_bytes(self.name.as_bytes());
        ser.write_bool(self.collapsed);
    }

    pub fn deserialize<T: Deserializer>(des: &mut T) -> Self {
        Self {
            id: CollectionId::from_raw(des.read_u32_be()),
            name: {
                let len = des.read_u32_be();
                let data = des.read_bytes(len as usize);
                String::from_utf8_lossy(&data).into_owned()
            },
            collapsed: des.read_bool(),
            rename: RenameState::None,
        }
    }
}

impl PostProcessing {
    pub fn serialize<T: Serializer>(&self, ser: &mut T) {
        self.variable_layer_height.serialize(ser);
        self.elephant_foot_fixer.serialize(ser);
    }

    pub fn deserialize<T: Deserializer>(des: &mut T, version: u16) -> Self {
        (version < 5).then(|| des.advance_by(5));
        Self {
            variable_layer_height: if version < 10 {
                Default::default()
            } else {
                VariableLayerHeight::deserialize(des, version)
            },
            elephant_foot_fixer: ElephantFootFixer::deserialize(des),
        }
    }
}

fn serialize_mesh_inner<T: Serializer>(ser: &mut T, mesh: &Arc<MeshInner>, progress: &Progress) {
    ser.write_u32_be(mesh.vertices.len() as u32);
    for vert in mesh.vertices.iter() {
        vert.serialize(ser);
        progress.add_complete(1);
    }

    ser.write_u32_be(mesh.faces.len() as u32);
    for face in mesh.faces.iter() {
        ser.write_u32_be(face[0]);
        ser.write_u32_be(face[1]);
        ser.write_u32_be(face[2]);
        progress.add_complete(3);
    }
}

fn deserialize_mesh_inner<T: Deserializer>(des: &mut T, progress: &Progress) -> MeshInner {
    let verts = des.read_u32_be();
    let verts = (0..verts)
        .map(|_| Vector3::<f32>::deserialize(des))
        .inspect(|_| progress.add_complete(4 * 3))
        .collect::<Vec<_>>();

    let faces = des.read_u32_be();
    let faces = (0..faces)
        .map(|_| [des.read_u32_be(), des.read_u32_be(), des.read_u32_be()])
        .inspect(|_| progress.add_complete(4 * 3))
        .collect::<Vec<_>>();

    MeshInner {
        vertices: verts.into_boxed_slice(),
        faces: faces.into_boxed_slice(),
    }
}

#[cfg(test)]
mod tests {
    use common::{
        serde::{DynamicSerializer, SliceDeserializer},
        units::Milimeters,
    };
    use nalgebra::Vector3;
    use slicer::{
        geometry::mesh::Mesh,
        post_process::hollowing::{DrainHole, HollowConfig},
    };

    use super::{ModelInfo, VERSION};
    use crate::core::project::model::Model;

    fn round_trip(model: &Model, version: u16) -> ModelInfo {
        let mut ser = DynamicSerializer::new();
        ModelInfo::new(0, model).serialize(&mut ser);
        let data = ser.into_inner();
        ModelInfo::deserialize(&mut SliceDeserializer::new(&data), version)
    }

    #[test]
    fn hollowing_settings_are_saved() {
        let hollow = HollowConfig {
            enabled: true,
            wall_thickness: Milimeters::new(1.5),
        };
        let holes = vec![
            DrainHole {
                position: Vector3::new(1.0, 2.0, 3.0),
                normal: Vector3::new(0.0, 0.0, 1.0),
                radius: 1.5,
            },
            DrainHole {
                position: Vector3::new(-4.0, 5.5, 0.25),
                normal: Vector3::new(1.0, 0.0, 0.0),
                radius: 0.75,
            },
        ];
        let model = Model::from_mesh(Mesh::default())
            .with_hollow(hollow)
            .with_drain_holes(holes.clone());

        let info = round_trip(&model, VERSION);
        assert_eq!(info.hollow, hollow);
        assert_eq!(info.drain_holes, holes);
        assert_eq!(info.exposure, model.exposure);
    }
}

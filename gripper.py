"""Ersatzgreifer mit originaler aeusserer Finray-Geometrie.

Die strukturierte Nachgiebigkeit wird am gesamten Greifrahmen als 6D-D6-
Ersatzmodell abgebildet. Der bereits gegriffene Stecker ist ueber ein
FixedJoint mit diesem Rahmen verbunden. Damit wird nicht gleichzeitig ein
unvalidiertes Fingerkontaktmodell als zusaetzlicher Versuchsparameter
eingefuehrt.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from contact_materials import bind_physics_material
from obj_loader import author_obj_mesh
from project_config import CONTACT, TASKS
from robot_visual import _axis_angle_matrix, mjcf_body_transform
from usd_body import UsdRigidBodyHandle


def _quat_rotate_wxyz(quaternion, vector):
    w, x, y, z = np.asarray(quaternion, dtype=float)
    qv = np.array([x, y, z])
    vector = np.asarray(vector, dtype=float)
    return (
        2.0 * np.dot(qv, vector) * qv
        + (w * w - np.dot(qv, qv)) * vector
        + 2.0 * w * np.cross(qv, vector)
    )


def _box(
    stage,
    prim_path,
    translation,
    size,
    color,
    collision=False,
    material=None,
):
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    xform.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        bind_physics_material(stage, prim_path, material)
    return cube


def _euler_xyz_matrix(euler_string):
    """MuJoCo-Standard-Eulerfolge ``xyz`` als homogene Matrix."""

    if not euler_string:
        return np.eye(4)
    x, y, z = (float(value) for value in euler_string.split())
    return (
        _axis_angle_matrix((1.0, 0.0, 0.0), x)
        @ _axis_angle_matrix((0.0, 1.0, 0.0), y)
        @ _axis_angle_matrix((0.0, 0.0, 1.0), z)
    )


def _mesh_transform_relative_to_tcp(
    mjcf_path,
    body_name,
    mesh_name,
    joint_positions,
):
    """Liest Body-, Slide- und Geom-Pose direkt aus ``robot_15.xml``."""

    root = ET.parse(mjcf_path).getroot()
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"MJCF-Koerper nicht gefunden: {body_name}")
    geom = next(
        (
            candidate
            for candidate in body.findall("geom")
            if candidate.get("mesh") == mesh_name
            and candidate.get("class") == "visual"
        ),
        None,
    )
    if geom is None:
        raise ValueError(
            f"Visual-Mesh {mesh_name} in MJCF-Koerper {body_name} nicht gefunden."
        )
    tcp = mjcf_body_transform(
        mjcf_path,
        "TCP",
        joint_positions=joint_positions,
    )
    body_transform = mjcf_body_transform(
        mjcf_path,
        body_name,
        joint_positions=joint_positions,
    )
    return (
        np.linalg.inv(tcp)
        @ body_transform
        @ _euler_xyz_matrix(geom.get("euler"))
    )


def original_gripper_visual_transforms(task_id, mjcf_path=None):
    """Exakte Halter-/Fingerposen aus der MuJoCo-Hierarchie.

    Der PhysX-``GraspFrame`` liegt am Steckerzentrum. Die MJCF-Geometrien
    werden zuerst relativ zum MuJoCo-TCP berechnet und danach um den
    aufgabenspezifischen TCP-Stecker-Abstand verschoben. Damit bleibt die
    Greiferstruktur fuer KET12 und USB identisch; nur Greifweite und Lage des
    jeweils gehaltenen Steckers unterscheiden sich.
    """

    if mjcf_path is None:
        mjcf_path = (
            Path(__file__).resolve().parent
            / "reference_mujoco"
            / "robot_15.xml"
        )
    mjcf_path = Path(mjcf_path)
    task = TASKS[task_id]
    closed = task.gripper_closed_axis_position_m
    joint_positions = {
        "gripper_axis_left_joint": -closed,
        "gripper_axis_right_joint": closed,
    }
    peg_to_tcp = np.array(
        [0.0, 0.0, task.tcp_offset_above_peg_center_m],
        dtype=float,
    )
    definitions = {
        "mount_left": ("gripper_mount_left", "gripper_mount_left"),
        "mount_right": ("gripper_mount_right", "gripper_mount_right"),
        "finray_left": (
            "finray_gripper_left",
            "finray_gripper_15_visual",
        ),
        "finray_right": (
            "finray_gripper_right",
            "finray_gripper_15_visual",
        ),
    }
    result = {}
    for label, (body_name, mesh_name) in definitions.items():
        transform = _mesh_transform_relative_to_tcp(
            mjcf_path,
            body_name,
            mesh_name,
            joint_positions,
        )
        result[label] = {
            "translation": tuple(transform[:3, 3] + peg_to_tcp),
            "basis": tuple(tuple(row) for row in transform[:3, :3]),
        }
    return result


def finray_visual_transforms(task_id, mjcf_path=None):
    """Kompatibler Zugriff nur auf die beiden exakten Finray-Posen."""

    transforms = original_gripper_visual_transforms(task_id, mjcf_path)
    return {
        "left": transforms["finray_left"],
        "right": transforms["finray_right"],
    }


class GripperAssembly:
    def __init__(
        self,
        stage,
        root_path,
        task_id,
        initial_peg_center,
        gripper_material,
        asset_dir: Path,
        show_original_visuals: bool,
    ):
        from pxr import Gf, UsdGeom, UsdPhysics

        self.task = TASKS[task_id]
        self.root_path = root_path
        self.reference_prim_path = f"{root_path}/TCPReference"
        self.reference = UsdGeom.Xform.Define(stage, self.reference_prim_path)
        self.reference_translate = self.reference.AddTranslateOp(
            UsdGeom.XformOp.PrecisionDouble
        )
        self.reference_orient = self.reference.AddOrientOp(
            UsdGeom.XformOp.PrecisionFloat
        )
        self.reference_orient.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        UsdPhysics.RigidBodyAPI.Apply(
            self.reference.GetPrim()
        ).CreateKinematicEnabledAttr(True)

        self.frame = UsdRigidBodyHandle(
            stage,
            f"{root_path}/GraspFrame",
            initial_peg_center,
            # Zwei Finray-Finger mit je 0.002 kg laut robot_15.xml.
            mass_kg=0.004,
        )
        self.grasp_frame_prim_path = self.frame.prim_path
        peg_x, peg_y, peg_z = self.task.peg_size_xyz_m
        pad_thickness = 0.0015
        pad_height = 0.010
        interference = 0.0001
        pad_x = peg_x / 2.0 + pad_thickness / 2.0 - interference
        grasp_z = peg_z / 2.0 - 0.010

        mount_left = asset_dir / "gripper_mount_left.obj"
        mount_right = asset_dir / "gripper_mount_right.obj"
        mjcf_path = asset_dir.parent.parent / "reference_mujoco" / "robot_15.xml"
        use_original_mounts = (
            show_original_visuals
            and mount_left.exists()
            and mount_right.exists()
            and mjcf_path.exists()
        )
        if use_original_mounts:
            visual = original_gripper_visual_transforms(
                task_id,
                mjcf_path,
            )
            author_obj_mesh(
                stage,
                f"{self.frame.prim_path}/GripperMountLeftVisual",
                mount_left,
                color=(0.28, 0.29, 0.32),
                **visual["mount_left"],
            )
            author_obj_mesh(
                stage,
                f"{self.frame.prim_path}/GripperMountRightVisual",
                mount_right,
                color=(0.28, 0.29, 0.32),
                **visual["mount_right"],
            )
        else:
            _box(
                stage,
                f"{self.frame.prim_path}/Carrier",
                (0.0, 0.0, self.task.tcp_offset_above_peg_center_m),
                (
                    max(0.050, peg_x + 0.030),
                    max(0.025, peg_y + 0.010),
                    0.008,
                ),
                (0.20, 0.21, 0.25),
            )
            for label, sign in (("Left", -1.0), ("Right", 1.0)):
                _box(
                    stage,
                    f"{self.frame.prim_path}/{label}ContactPad",
                    (sign * pad_x, 0.0, grasp_z),
                    (pad_thickness, max(0.010, peg_y * 0.8), pad_height),
                    (0.08, 0.08, 0.09),
                    collision=False,
                )

        if show_original_visuals:
            mesh = asset_dir / "finray_gripper_15_visual.obj"
            if mesh.exists() and mjcf_path.exists():
                # Die CAD-Datei ist das unverformte aeussere Finray-Modell.
                # Physik bleibt fuer die 100er-Studie beim D6-Ersatzmodell.
                visual = finray_visual_transforms(task_id, mjcf_path)
                author_obj_mesh(
                    stage,
                    f"{self.frame.prim_path}/FinrayLeftVisual",
                    mesh,
                    color=(0.95, 0.55, 0.08),
                    **visual["left"],
                )
                author_obj_mesh(
                    stage,
                    f"{self.frame.prim_path}/FinrayRightVisual",
                    mesh,
                    color=(0.95, 0.55, 0.08),
                    **visual["right"],
                )

        self.set_tcp_pose(
            np.asarray(initial_peg_center)
            + np.array([0.0, 0.0, self.task.tcp_offset_above_peg_center_m]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def attach_peg(self, stage, peg_prim_path):
        """Bindet den bereits gegriffenen Stecker stabil an den Greifrahmen."""

        from pxr import Gf, Sdf, UsdPhysics

        self.grasp_joint_prim_path = f"{self.root_path}/HeldPegJoint"
        joint = UsdPhysics.FixedJoint.Define(
            stage,
            self.grasp_joint_prim_path,
        )
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.grasp_frame_prim_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(peg_prim_path)])
        zero = Gf.Vec3f(0.0, 0.0, 0.0)
        identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos0Attr().Set(zero)
        joint.CreateLocalPos1Attr().Set(zero)
        joint.CreateLocalRot0Attr().Set(identity)
        joint.CreateLocalRot1Attr().Set(identity)
        joint.CreateCollisionEnabledAttr(False)
        return joint

    def set_tcp_pose(self, tcp_position, orientation_wxyz):
        from pxr import Gf

        orientation = np.asarray(orientation_wxyz, dtype=float)
        offset_world = _quat_rotate_wxyz(
            orientation,
            (0.0, 0.0, self.task.tcp_offset_above_peg_center_m),
        )
        reference_position = np.asarray(tcp_position, dtype=float) - offset_world
        self.reference_translate.Set(Gf.Vec3d(*map(float, reference_position)))
        self.reference_orient.Set(
            Gf.Quatf(
                float(orientation[0]),
                Gf.Vec3f(*map(float, orientation[1:])),
            )
        )

    def get_reference_position(self):
        return np.array(self.reference_translate.Get(), dtype=float)

    def get_grasp_frame_position(self):
        return self.frame.get_world_pose()[0]

    def get_grasp_frame_orientation(self):
        return self.frame.get_world_pose()[1]

    def reset(self, tcp_position, orientation_wxyz):
        self.set_tcp_pose(tcp_position, orientation_wxyz)
        reference_position = self.get_reference_position()
        self.frame.set_world_pose(reference_position, orientation_wxyz)
        self.frame.set_linear_velocity(np.zeros(3))
        self.frame.set_angular_velocity(np.zeros(3))

    def grip_error(self, peg_position):
        return np.asarray(peg_position) - self.get_grasp_frame_position()

    @staticmethod
    def nominal_pad_normal_force_n():
        return 2.0 * CONTACT.gripper_contact_stiffness_n_per_m * 0.0001

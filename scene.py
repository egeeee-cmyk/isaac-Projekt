"""Aufbau vieler unabhaengiger Montageumgebungen in einer PhysX-Szene."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compliance import StructuredComplianceJoint
from contact_materials import bind_physics_material, create_environment_materials
from gripper import GripperAssembly
from motion import quaternion_x_deg, target_pose
from project_config import TASKS, EnvironmentParameters


RENDER_CAMERA_PATH = "/World/RenderCamera"
# MuJoCo-Meshes sind entlang ihrer lokalen Y-Achse gesteckt. In der
# Isaac-Studie ist Z die Einsetzachse: (x, y, z)_OBJ -> (x, -z, y)_Isaac.
INSERTION_Y_TO_Z_BASIS = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
)
KET12_NIST_HOLE_ANCHOR_M = (-0.0415, 0.0290, -0.1971)
USB_FEMALE_MOUTH_ANCHOR_M = (0.0, 0.0400, 0.0)


@dataclass
class EnvironmentRuntime:
    parameters: EnvironmentParameters
    root_path: str
    origin: np.ndarray
    socket_center_xy: np.ndarray
    peg: object
    gripper: GripperAssembly
    compliance_joint: StructuredComplianceJoint


def create_render_environment(stage):
    """Erzeugt eigene Beleuchtung und eine feste Präsentationskamera.

    Die Viewport-Standardbeleuchtung ist eine persistente Isaac-UI-Einstellung.
    Steht sie auf ``Stage Lights`` und enthält die Szene keine Lichter, bleibt
    der Viewport vollständig schwarz. Deshalb ist die sichtbare Debugszene
    unabhängig von lokalen Isaac-Einstellungen.
    """

    from pxr import Gf, UsdGeom, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.GetIntensityAttr().Set(450.0)
    dome.GetColorAttr().Set(Gf.Vec3f(0.88, 0.92, 1.00))

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/KeyLight")
    key.GetIntensityAttr().Set(1100.0)
    key.GetColorAttr().Set(Gf.Vec3f(1.00, 0.96, 0.90))
    key.GetAngleAttr().Set(1.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-35.0, 25.0, 25.0)
    )

    camera = UsdGeom.Camera.Define(stage, RENDER_CAMERA_PATH)
    camera.GetFocalLengthAttr().Set(32.0)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.001, 100.0))
    return RENDER_CAMERA_PATH


def _fixed_box(world, FixedCuboid, path, name, position, size, color, material):
    box = world.scene.add(
        FixedCuboid(
            prim_path=path,
            name=name,
            position=np.asarray(position, dtype=float),
            scale=np.asarray(size, dtype=float),
            color=np.asarray(color, dtype=float),
        )
    )
    if material is not None:
        import omni.usd

        bind_physics_material(omni.usd.get_context().get_stage(), path, material)
    return box


def _set_display_opacity(stage, prim_path, opacity):
    """Blendet nur die Darstellung aus; Kollision und Dynamik bleiben aktiv."""

    from pxr import UsdGeom

    geometry = UsdGeom.Gprim(stage.GetPrimAtPath(prim_path))
    geometry.CreateDisplayOpacityAttr().Set([float(opacity)])


def _author_original_socket_visual(
    stage,
    root,
    task_id,
    asset_dir,
    socket_center,
    socket_top_z,
):
    """Zeigt die originale MuJoCo-Aufnahme über der robusten Ersatzkollision."""

    from obj_loader import author_obj_mesh

    if task_id == "USB":
        mesh_path = asset_dir / "USB_Female.obj"
        anchor = USB_FEMALE_MOUTH_ANCHOR_M
    else:
        mesh_path = asset_dir / "NIST_Board_Lifted.obj"
        anchor = KET12_NIST_HOLE_ANCHOR_M
    if not mesh_path.exists():
        return False
    author_obj_mesh(
        stage,
        f"{root}/OriginalSocketVisual",
        mesh_path,
        color=(0.72, 0.74, 0.78),
        translation=(
            float(socket_center[0]),
            float(socket_center[1]),
            float(socket_top_z),
        ),
        basis=INSERTION_Y_TO_Z_BASIS,
        anchor=anchor,
    )
    return True


def _author_original_peg_visual(stage, peg, task_id, asset_dir):
    """Legt das USB-Male-OBJ auf den physikalisch robusten Ersatzkörper."""

    if task_id != "USB":
        # Richards MuJoCo-Basis modelliert KET12 selbst als Box; dafür gibt es
        # kein separates KET12-CAD-Mesh, das hier vorgetäuscht werden dürfte.
        return False
    from obj_loader import author_obj_mesh, obj_bounds

    mesh_path = asset_dir / "USB_Male.obj"
    if not mesh_path.exists():
        return False
    _, _, center = obj_bounds(str(mesh_path.resolve()))
    author_obj_mesh(
        stage,
        f"{peg.prim_path}/OriginalUSBMaleVisual",
        mesh_path,
        color=(0.32, 0.66, 0.76),
        basis=INSERTION_Y_TO_Z_BASIS,
        anchor=center,
    )
    return True


def create_environment(
    world,
    stage,
    DynamicCuboid,
    FixedCuboid,
    parameters: EnvironmentParameters,
    origin,
    asset_dir: Path,
    show_original_visuals=False,
):
    from pxr import UsdGeom

    task = TASKS[parameters.task_id]
    env_id = parameters.env_id
    root = f"/World/Envs/Env_{env_id:03d}"
    UsdGeom.Xform.Define(stage, root)
    origin = np.asarray(origin, dtype=float)
    materials = create_environment_materials(stage, root, parameters)

    _fixed_box(
        world,
        FixedCuboid,
        f"{root}/Workplate",
        f"env_{env_id:03d}_workplate",
        origin + np.array([0.0, 0.0, -0.006]),
        (0.28, 0.24, 0.012),
        (0.42, 0.58, 0.72),
        None,
    )
    socket_center = origin[:2] + np.array(
        [parameters.socket_shift_x_m, parameters.socket_shift_y_m]
    )
    hole_x, hole_y = task.hole_size_xy_m
    wall = task.socket_wall_thickness_m
    height = task.socket_wall_height_m
    z = height / 2.0
    socket_defs = (
        (
            "Front",
            (0.0, hole_y / 2.0 + wall / 2.0, z),
            (hole_x + 2.0 * wall, wall, height),
        ),
        (
            "Back",
            (0.0, -hole_y / 2.0 - wall / 2.0, z),
            (hole_x + 2.0 * wall, wall, height),
        ),
        (
            "Right",
            (hole_x / 2.0 + wall / 2.0, 0.0, z),
            (wall, hole_y, height),
        ),
        (
            "Left",
            (-hole_x / 2.0 - wall / 2.0, 0.0, z),
            (wall, hole_y, height),
        ),
    )
    socket_boxes = []
    for label, local_position, size in socket_defs:
        world_position = np.array(
            [
                socket_center[0] + local_position[0],
                socket_center[1] + local_position[1],
                local_position[2],
            ]
        )
        socket_box = _fixed_box(
            world,
            FixedCuboid,
            f"{root}/Socket/{label}",
            f"env_{env_id:03d}_socket_{label.lower()}",
            world_position,
            size,
            (0.68, 0.70, 0.74),
            materials["socket"],
        )
        socket_boxes.append(socket_box)

    if show_original_visuals and _author_original_socket_visual(
        stage,
        root,
        parameters.task_id,
        asset_dir,
        socket_center,
        height,
    ):
        for socket_box in socket_boxes:
            _set_display_opacity(stage, socket_box.prim_path, 0.0)

    first_target = target_pose(parameters.task_id, 0.0)
    first_orientation = quaternion_x_deg(first_target.tilt_x_deg)
    first_tcp = origin + first_target.position_xyz_m
    first_peg_center = first_tcp - np.array(
        [0.0, 0.0, task.tcp_offset_above_peg_center_m]
    )
    peg = world.scene.add(
        DynamicCuboid(
            prim_path=f"{root}/Peg",
            name=f"env_{env_id:03d}_peg",
            position=first_peg_center,
            # Fuer USB bildet die Kollisionsbreite die Metallspitze ab, nicht
            # die groessere Kunststoff-/Greifkontur des Gesamtsteckers.
            scale=np.asarray(
                (
                    task.peg_insertion_size_xy_m[0],
                    task.peg_insertion_size_xy_m[1],
                    task.peg_size_xyz_m[2],
                )
            ),
            color=np.array([0.10, 0.30, 1.00])
            if parameters.task_id == "KET12"
            else np.array([0.20, 0.72, 0.82]),
            mass=task.peg_mass_kg,
        )
    )
    bind_physics_material(stage, peg.prim_path, materials["peg"])
    if show_original_visuals and _author_original_peg_visual(
        stage,
        peg,
        parameters.task_id,
        asset_dir,
    ):
        _set_display_opacity(stage, peg.prim_path, 0.0)

    gripper = GripperAssembly(
        stage,
        f"{root}/Gripper",
        parameters.task_id,
        first_peg_center,
        materials["gripper"],
        asset_dir,
        show_original_visuals,
    )
    compliance_joint = StructuredComplianceJoint(
        stage,
        f"{root}/Gripper",
        gripper.reference_prim_path,
        gripper.grasp_frame_prim_path,
        parameters,
    )
    gripper.attach_peg(stage, peg.prim_path)
    return EnvironmentRuntime(
        parameters=parameters,
        root_path=root,
        origin=origin,
        socket_center_xy=socket_center,
        peg=peg,
        gripper=gripper,
        compliance_joint=compliance_joint,
    )


def reset_environment(environment: EnvironmentRuntime):
    """Setzt Peg und Ersatzgreifer reproduzierbar auf den ersten Sollzustand."""

    task = TASKS[environment.parameters.task_id]
    target = target_pose(environment.parameters.task_id, 0.0)
    orientation = quaternion_x_deg(target.tilt_x_deg)
    tcp_position = environment.origin + target.position_xyz_m
    offset_world = np.array(
        [0.0, 0.0, task.tcp_offset_above_peg_center_m],
        dtype=float,
    )
    peg_center = tcp_position - offset_world
    environment.peg.set_world_pose(
        position=peg_center,
        orientation=orientation,
    )
    environment.peg.set_linear_velocity(np.zeros(3))
    environment.peg.set_angular_velocity(np.zeros(3))
    environment.gripper.reset(tcp_position, orientation)


def set_target(environment: EnvironmentRuntime, time_s):
    target = target_pose(environment.parameters.task_id, time_s)
    orientation = quaternion_x_deg(target.tilt_x_deg)
    environment.gripper.set_tcp_pose(
        environment.origin + target.position_xyz_m,
        orientation,
    )
    return target.phase

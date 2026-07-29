"""Direkter Visual-Import des UR5e aus dem MuJoCo-MJCF.

Der Import ist absichtlich rein visuell. Die 100er-Physik wird weiterhin vom
vorgegebenen TCP angetrieben, damit Roboterkinematik nicht als zusaetzlicher
Versuchsparameter in die Nachgiebigkeitsstudie eingeht.

Der UR5e wird nicht mehr nur mit der Home-Konfiguration neben die Aufgabe
gestellt. Fuer beide Aufgaben sind reproduzierbare IK-Posen fuer alle
TCP-Wegpunkte hinterlegt. Ihre TCP-Position wird aus derselben MJCF-Hierarchie
berechnet, die fuer das Mesh-Authoring verwendet wird. Zwischen den
Wegpunkten werden die Gelenkwinkel zeitgleich zur Montagebahn interpoliert.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from obj_loader import author_obj_mesh


_HOME_JOINTS_RAD = {
    "shoulder_pan_joint": 3.0 * np.pi / 2.0,
    "shoulder_lift_joint": -np.pi / 2.0,
    "elbow_joint": -np.pi / 2.0,
    "wrist_1_joint": 3.0 * np.pi / 2.0,
    "wrist_2_joint": np.pi / 2.0,
    "wrist_3_joint": 0.0,
}

# Vorgegebene Debugpose der UR5e-Basis relativ zur lokalen Buchsenmitte.
# Sie entspricht der Basislage der gelieferten MuJoCo-Szene.
_BASE_TRANSLATION_BY_TASK = {
    "KET12": np.array([0.3414, -0.3971, 0.0]),
    "USB": np.array([0.3414, -0.4722, 0.0]),
}

# Mit der MJCF-Kette numerisch geloeste Posen fuer die ersten Debug-Soll-TCPs.
# Zielorientierung ist jeweils [0, 0, 0] Grad wie in reference_mujoco/sim.py.
_PRESENTATION_JOINT_NAMES = tuple(_HOME_JOINTS_RAD)
_PRESENTATION_TIMES_S_BY_TASK = {
    "KET12": np.array(
        [0.0, 1.0, 2.5, 3.1, 3.7, 4.3, 5.0, 6.3, 7.0, 7.5],
        dtype=float,
    ),
    "USB": np.array(
        [0.0, 1.0, 2.5, 3.1, 3.7, 4.3, 5.0, 5.5, 5.9, 6.3, 7.0, 7.5],
        dtype=float,
    ),
}
_PRESENTATION_TRAJECTORY_RAD = {
    "KET12": np.array(
        [
            [4.10861336418202, -1.84354152569104, -2.02080667301074, 5.43514452549668, 1.5707963267949, -0.603775616202671],
            [4.10861336418202, -1.84354152569104, -2.02080667301074, 5.43514452549668, 1.5707963267949, -0.603775616202671],
            [4.13720211099562, -1.89460664807652, -2.09281436087310, 5.60577454818045, 1.49759941218776, -0.576928495349718],
            [4.13720211099562, -1.90298661930557, -2.09586746351993, 5.61720762205632, 1.49759941218776, -0.576928495349717],
            [4.13499759142170, -1.90056151061368, -2.09966191603795, 5.61873831918860, 1.49770439117298, -0.579136417641785],
            [4.13665225904043, -1.90444031657766, -2.09754576994927, 5.62037989241386, 1.49762556281664, -0.577479199191005],
            [4.14863212206989, -1.89468427883616, -2.11220858161792, 5.62440609135222, 1.49706082868777, -0.565480299223424],
            [4.14836627524588, -1.94583679471854, -2.13172004925434, 5.70193866426581, 1.49773204327116, -0.56548029922342],
            [4.11014105739351, -1.96181221889925, -2.08190075393612, 5.62104751797767, 1.57163436899567, -0.60192016789942],
            [4.11014105739351, -1.96181221889925, -2.08190075393612, 5.62104751797767, 1.57163436899567, -0.60192016789942],
        ],
        dtype=float,
    ),
    "USB": np.array(
        [
            [4.001236332226489, -1.898088545831711, -1.816244701469052, 5.285129574095660, 1.570796326794897, -0.711152648158201],
            [4.001236332226489, -1.898088545831711, -1.816244701469052, 5.285129574095660, 1.570796326794897, -0.711152648158201],
            [4.049400776250313, -1.916153452972768, -1.946199922351912, 5.541251440367855, 1.433503464663229, -0.670427893232264],
            [4.052366062679881, -1.935777504282935, -1.950111446079082, 5.564379227864657, 1.433184136852896, -0.667451781004783],
            [4.046645423895816, -1.930011301747539, -1.959164255972645, 5.568451662526773, 1.433801261215401, -0.673193069147059],
            [4.052366062679881, -1.935777504282935, -1.950111446079082, 5.564379227864657, 1.433184136852896, -0.667451781004784],
            [4.067204913137267, -1.920344814138010, -1.972814649887997, 5.569594508252907, 1.431604301321281, -0.652554816379155],
            [4.067204913098814, -1.920344814470525, -1.972814648359400, 5.569594504633408, 1.431604301429570, -0.652554816080792],
            [4.051372837379388, -1.930689129788970, -1.957909487848057, 5.567225591384849, 1.433290961380068, -0.668448660809613],
            [3.999632824098012, -2.025318849943632, -1.889705197987355, 5.485820364359284, 1.570796327105705, -0.712756156286678],
            [3.999632824106877, -2.060855048653492, -1.902120493404348, 5.533771858659459, 1.570796327080326, -0.712756156277813],
            [3.999632824106877, -2.060855048653492, -1.902120493404348, 5.533771858659459, 1.570796327080326, -0.712756156277813],
        ],
        dtype=float,
    ),
}

_COLORS = {
    "black": (0.033, 0.033, 0.033),
    "jointgray": (0.278, 0.278, 0.278),
    "linkgray": (0.82, 0.82, 0.82),
    "urblue": (0.49, 0.678, 0.80),
}

# Der physische Ersatzgreifer wird separat am bewegten GraspFrame aufgebaut.
# Diese MJCF-Unterbaeume duerfen deshalb nicht ein zweites Mal am
# Roboter erscheinen.
_SKIPPED_VISUAL_SUBTREES = {"TCP", "gripper_mount_mount"}


def _numbers(value, default):
    if not value:
        return default
    return tuple(float(item) for item in value.split())


def _normalize_quaternion_wxyz(quaternion):
    quaternion = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("MJCF-Quaternion hat Laenge null.")
    return quaternion / norm


def _quaternion_matrix_wxyz(quaternion):
    """Homogene Rotationsmatrix fuer ein MJCF-Quaternion ``w x y z``."""

    w, x, y, z = _normalize_quaternion_wxyz(quaternion)
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return matrix


def _axis_angle_matrix(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("MJCF-Gelenkachse hat Laenge null.")
    x, y, z = axis / norm
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_minus_c = 1.0 - c
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=float,
    )
    return matrix


def _xyaxes_matrix(value):
    values = np.asarray(_numbers(value, None), dtype=float)
    if values.shape != (6,):
        raise ValueError(f"Ungueltiges MJCF-xyaxes-Attribut: {value}")
    x_axis = values[:3]
    y_axis = values[3:]
    x_axis /= np.linalg.norm(x_axis)
    y_axis -= np.dot(x_axis, y_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    return matrix


def _body_local_matrix(body, joint_positions=None):
    matrix = np.eye(4)
    matrix[:3, 3] = _numbers(body.get("pos"), (0.0, 0.0, 0.0))
    quaternion = _numbers(body.get("quat"), None)
    if quaternion is not None:
        matrix = matrix @ _quaternion_matrix_wxyz(quaternion)
    elif body.get("xyaxes") is not None:
        matrix = matrix @ _xyaxes_matrix(body.get("xyaxes"))
    positions = _HOME_JOINTS_RAD if joint_positions is None else joint_positions
    # Ein MJCF-Body darf mehrere Gelenke enthalten. Fuer die UR5e-Kette ist
    # es jeweils ein Hinge-Joint; die Greiferachsen sind Slide-Joints. Die
    # fruehere Implementierung behandelte jedes gefundene Gelenk als Rotation
    # und konnte deshalb die MuJoCo-Greiferhierarchie nicht exakt auswerten.
    for joint in body.findall("joint"):
        joint_name = joint.get("name")
        if joint_name not in positions:
            continue
        axis = _numbers(joint.get("axis"), (0.0, 1.0, 0.0))
        value = float(positions[joint_name])
        if joint.get("type", "hinge") == "slide":
            axis = np.asarray(axis, dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                raise ValueError("MJCF-Gelenkachse hat Laenge null.")
            translation = np.eye(4)
            translation[:3, 3] = axis / norm * value
            matrix = matrix @ translation
        else:
            matrix = matrix @ _axis_angle_matrix(axis, value)
    return matrix


def mjcf_body_transform(
    mjcf_path: Path,
    body_name: str,
    joint_positions=None,
):
    """Berechnet die lokale Home-Pose eines MJCF-Koerpers.

    Die Funktion ist USD-unabhaengig und wird deshalb auch in den
    Kernlogiktests verwendet.
    """

    root = ET.parse(mjcf_path).getroot()
    base = next(
        body
        for body in root.findall("./worldbody/body")
        if body.get("name") == "base"
    )

    def visit(body, parent_matrix):
        world_matrix = parent_matrix @ _body_local_matrix(
            body,
            joint_positions=joint_positions,
        )
        if body.get("name") == body_name:
            return world_matrix
        for child in body.findall("body"):
            result = visit(child, world_matrix)
            if result is not None:
                return result
        return None

    result = visit(base, np.eye(4))
    if result is None:
        raise ValueError(f"MJCF-Koerper nicht gefunden: {body_name}")
    return result


def robot_translation_for_tcp(
    mjcf_path: Path,
    desired_tcp_world,
    joint_positions=None,
):
    """Welttranslation, die den Home-TCP auf ``desired_tcp_world`` legt."""

    local_tcp = mjcf_body_transform(
        mjcf_path,
        "TCP",
        joint_positions=joint_positions,
    )[:3, 3]
    return np.asarray(desired_tcp_world, dtype=float) - local_tcp


def presentation_base_translation(task_id, environment_origin):
    if task_id not in _BASE_TRANSLATION_BY_TASK:
        raise ValueError(f"Keine UR5e-Debugbasis fuer Aufgabe {task_id}")
    return (
        np.asarray(environment_origin, dtype=float)
        + _BASE_TRANSLATION_BY_TASK[task_id]
    )


def presentation_joint_positions(task_id):
    return presentation_joint_positions_at(task_id, 0.0)


def presentation_joint_positions_at(task_id, time_s):
    if task_id not in _PRESENTATION_TRAJECTORY_RAD:
        raise ValueError(f"Keine UR5e-Debugpose fuer Aufgabe {task_id}")
    trajectory = _PRESENTATION_TRAJECTORY_RAD[task_id]
    times = _PRESENTATION_TIMES_S_BY_TASK[task_id]
    time_s = float(np.clip(time_s, times[0], times[-1]))
    right = int(np.searchsorted(times, time_s, side="right"))
    if right == 0:
        values = trajectory[0]
    elif right >= len(times):
        values = trajectory[-1]
    else:
        left = right - 1
        t0 = times[left]
        t1 = times[right]
        alpha = (time_s - t0) / (t1 - t0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        values = trajectory[left] * (1.0 - smooth) + trajectory[right] * smooth
    return dict(zip(_PRESENTATION_JOINT_NAMES, values))


def update_authored_joint_positions(placement, joint_positions):
    """Aktualisiert die rein visuellen UR5e-Gelenke fuer den Debuglauf."""

    for joint_name, value_rad in joint_positions.items():
        op = placement["joint_ops"].get(joint_name)
        if op is not None:
            op.Set(float(np.degrees(value_rad)))


def _apply_body_transform(xform, body):
    from pxr import Gf, UsdGeom

    position = _numbers(body.get("pos"), (0.0, 0.0, 0.0))
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    quaternion = _numbers(body.get("quat"), None)
    if quaternion is not None:
        q = _normalize_quaternion_wxyz(quaternion)
        xform.AddOrientOp().Set(
            Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:])))
        )


def _author_body(
    stage,
    parent_path,
    body,
    mesh_files,
    asset_dir,
    joint_positions=None,
    joint_ops=None,
    progress_callback=None,
):
    from pxr import Gf, UsdGeom

    name = body.get("name", "body").replace(" ", "_")
    if body.get("name") in _SKIPPED_VISUAL_SUBTREES:
        return
    body_path = f"{parent_path}/{name}"
    body_xform = UsdGeom.Xform.Define(stage, body_path)
    _apply_body_transform(body_xform, body)

    joint_parent = body_path
    joint = body.find("joint")
    positions = _HOME_JOINTS_RAD if joint_positions is None else joint_positions
    if joint is not None and joint.get("name") in positions:
        joint_name = joint.get("name")
        joint_path = f"{body_path}/JointPose"
        joint_xform = UsdGeom.Xform.Define(stage, joint_path)
        angle_deg = float(np.degrees(positions[joint_name]))
        axis = _numbers(joint.get("axis"), (0.0, 1.0, 0.0))
        if np.allclose(axis, (1.0, 0.0, 0.0)):
            rotate_op = joint_xform.AddRotateXOp()
        elif np.allclose(axis, (0.0, 0.0, 1.0)):
            rotate_op = joint_xform.AddRotateZOp()
        else:
            rotate_op = joint_xform.AddRotateYOp()
        rotate_op.Set(angle_deg)
        if joint_ops is not None:
            joint_ops[joint_name] = rotate_op
        joint_parent = joint_path

    visual_index = 0
    for geom in body.findall("geom"):
        if geom.get("class") != "visual" or not geom.get("mesh"):
            continue
        mesh_name = geom.get("mesh")
        file_name = mesh_files.get(mesh_name, f"{mesh_name}.obj")
        obj_path = asset_dir / file_name
        if not obj_path.exists():
            continue
        material = geom.get("material", "linkgray")
        author_obj_mesh(
            stage,
            f"{joint_parent}/Visual_{visual_index:02d}_{mesh_name}",
            obj_path,
            color=_COLORS.get(material, (0.75, 0.75, 0.78)),
        )
        visual_index += 1
        if progress_callback is not None:
            progress_callback()

    for child in body.findall("body"):
        _author_body(
            stage,
            joint_parent,
            child,
            mesh_files,
            asset_dir,
            joint_positions=joint_positions,
            joint_ops=joint_ops,
            progress_callback=progress_callback,
        )


def author_ur5e_finray_visual(
    stage,
    prim_path,
    mjcf_path: Path,
    asset_dir: Path,
    base_translation,
    joint_positions=None,
    progress_callback=None,
):
    """Importiert die UR5e-Linkmeshes und richtet sie am Soll-TCP aus.

    Der Name bleibt aus Kompatibilitaetsgruenden erhalten. Die Finray-Meshes
    werden ausschliesslich vom bewegten ``GripperAssembly`` erzeugt, damit
    keine statischen Dubletten mehr sichtbar sind.
    """

    from pxr import Gf, UsdGeom

    root = ET.parse(mjcf_path).getroot()
    mesh_files = {
        mesh.get("name", Path(mesh.get("file")).stem): mesh.get("file")
        for mesh in root.findall("./asset/mesh")
    }
    base = next(
        body for body in root.findall("./worldbody/body")
        if body.get("name") == "base"
    )
    translation = np.asarray(base_translation, dtype=float)
    container = UsdGeom.Xform.Define(stage, prim_path)
    container.AddTranslateOp().Set(Gf.Vec3d(*map(float, translation)))
    joint_ops = {}
    _author_body(
        stage,
        prim_path,
        base,
        mesh_files,
        asset_dir,
        joint_positions=joint_positions,
        joint_ops=joint_ops,
        progress_callback=progress_callback,
    )
    local_tcp = mjcf_body_transform(
        mjcf_path,
        "TCP",
        joint_positions=joint_positions,
    )[:3, 3]
    return {
        "prim_path": prim_path,
        "translation": translation,
        "tcp_world": translation + local_tcp,
        "local_tcp": local_tcp,
        "joint_ops": joint_ops,
    }

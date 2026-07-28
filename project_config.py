"""Zentrale, SI-konsistente Konfiguration der MuJoCo-zu-Isaac-Uebertragung.

Die MuJoCo-Referenz verwendet fuer die Finray-Finger lokale Achsen:
    x = compliance direction, y = assembly direction, z = depth direction.

Die parallele Isaac-Szene verwendet:
    X = compliance/lateral, Y = depth/lateral, Z = assembly/insertion.

Darum gilt die Achsenabbildung ``mj(x, y, z) -> isaac(X, Z, Y)``.
Alle Laengen sind Meter, Winkel Radiant, Zeiten Sekunden.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SolverConfig:
    """PhysX-Laufzeitkonfiguration.

    Der MuJoCo-Zeitschritt 5e-5 s bleibt als Referenz dokumentiert. Fuer
    hundert Umgebungen ist 1/1000 s der bewusst schnellere Studienwert.
    Der Einzelvergleich kann mit ``--physics-dt 5e-5`` gestartet werden.
    """

    study_physics_dt_s: float = 1.0 / 1000.0
    mujoco_reference_dt_s: float = 5.0e-5
    rendering_dt_s: float = 1.0 / 60.0
    total_time_s: float = 7.5
    position_iterations: int = 16
    velocity_iterations: int = 4
    gravity_m_per_s2: float = -9.81


@dataclass(frozen=True)
class MujocoCompliance:
    """Tatsaechliche Werte aus ``reference_mujoco/sim.py``.

    Die drei translatorischen Eingaben werden dort von N/mm nach N/m
    umgerechnet. Die Rotationen bleiben in N*m/rad.
    """

    compliance_stiffness_n_per_m: float = 1_200.0
    assembly_stiffness_n_per_m: float = 52_400.0
    depth_stiffness_n_per_m: float = 100_000.0
    rot_x_stiffness_nm_per_rad: float = 100.0
    rot_y_stiffness_nm_per_rad: float = 100.0
    rot_z_stiffness_nm_per_rad: float = 1.0
    translation_damping_ns_per_m: float = 10.0
    rotation_damping_nms_per_rad: float = 1.0


@dataclass(frozen=True)
class IsaacCompliance:
    """Auf Isaac-Weltachsen abgebildete D6-Parameter.

    Die hier lesbaren Winkelwerte bleiben in N*m/rad bzw. N*m*s/rad.
    ``compliance.py`` rechnet sie beim Authoring des USD-Drives mit pi/180
    in die vom USD-Schema verlangten Koeffizienten pro Grad um.
    """

    stiffness_x_n_per_m: float = 1_200.0
    stiffness_y_n_per_m: float = 100_000.0
    stiffness_z_n_per_m: float = 52_400.0
    damping_x_ns_per_m: float = 10.0
    damping_y_ns_per_m: float = 10.0
    damping_z_ns_per_m: float = 10.0

    # Achsenabbildung der Rotationen:
    # MuJoCo rot_x -> Isaac rotX
    # MuJoCo rot_z -> Isaac rotY
    # MuJoCo rot_y -> Isaac rotZ (Rotation um Einfuehrachse)
    stiffness_rot_x_nm_per_rad: float = 100.0
    stiffness_rot_y_nm_per_rad: float = 1.0
    stiffness_rot_z_nm_per_rad: float = 100.0
    damping_rot_x_nms_per_rad: float = 1.0
    damping_rot_y_nms_per_rad: float = 1.0
    damping_rot_z_nms_per_rad: float = 1.0

    max_linear_deflection_m: float = 0.030
    max_drive_force_n: float = 250.0
    max_drive_torque_nm: float = 20.0


@dataclass(frozen=True)
class ContactConfig:
    """PhysX-Ersatzparameter fuer die MuJoCo-Kontakte.

    MuJoCo ``friction`` enthaelt Gleit-, Torsions- und Rollreibung und ist
    deshalb kein statisch/dynamisch-Paar. Nur die Gleitkomponente kann direkt
    abgebildet werden. ``solref/solimp`` werden nicht blind in eine
    Kontaktsteifigkeit umgerechnet; die hier gesetzten PhysX-Werte sind
    explizite Kalibrierparameter fuer den Einzelvergleich.
    """

    peg_socket_static_friction: float = 0.20
    peg_socket_dynamic_friction: float = 0.20
    gripper_peg_static_friction_ket12: float = 1.40
    gripper_peg_dynamic_friction_ket12: float = 1.00
    gripper_peg_static_friction_usb: float = 1.00
    gripper_peg_dynamic_friction_usb: float = 1.00
    restitution: float = 0.0
    peg_socket_contact_stiffness_n_per_m: float = 1_000_000.0
    peg_socket_contact_damping_ns_per_m: float = 100.0
    gripper_contact_stiffness_n_per_m: float = 100_000.0
    gripper_contact_damping_ns_per_m: float = 10.0


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    peg_size_xyz_m: Tuple[float, float, float]
    # Effektiver Querschnitt des Bereichs, der tatsaechlich in die Buchse
    # eingefuehrt wird. Bei USB ist der Metallstecker deutlich schmaler als
    # das Gesamtgehaeuse; ein einziger Vollquader mit der Gehaeusebreite
    # verklemmt deshalb schon am Buchseneingang.
    peg_insertion_size_xy_m: Tuple[float, float]
    peg_mass_kg: float
    hole_size_xy_m: Tuple[float, float]
    socket_wall_height_m: float
    socket_wall_thickness_m: float
    socket_base_thickness_m: float
    tcp_offset_above_peg_center_m: float
    search_start_xyz_m: Tuple[float, float, float]
    search_tilt_deg: float
    # Inkremente entsprechen der festen MuJoCo-Suchstrategie.
    touch_z_delta_xyz_m: Tuple[float, float, float]
    touch_back_delta_xyz_m: Tuple[float, float, float]
    touch_front_delta_xyz_m: Tuple[float, float, float]
    touch_side_delta_xyz_m: Tuple[float, float, float]
    assembly_delta_xyz_m: Tuple[float, float, float]
    success_min_insertion_depth_m: float
    success_max_tilt_deg: float = 12.0
    evaluation_clearance_margin_m: float = 0.00025
    grip_loss_threshold_m: float = 0.004


TASKS: Dict[str, TaskConfig] = {
    "KET12": TaskConfig(
        task_id="KET12",
        # MuJoCo box size ist Halbbreite: 0.004 0.006 0.025.
        peg_size_xyz_m=(0.008, 0.012, 0.050),
        peg_insertion_size_xy_m=(0.008, 0.012),
        # Zwei identische MuJoCo-Boxgeoms (collision + visual) mit
        # Standarddichte 1000 kg/m^3 ergeben ca. 9.6 g.
        peg_mass_kg=0.0096,
        hole_size_xy_m=(0.0086, 0.0126),
        socket_wall_height_m=0.035,
        socket_wall_thickness_m=0.008,
        socket_base_thickness_m=0.005,
        tcp_offset_above_peg_center_m=0.025,
        search_start_xyz_m=(-0.0011, 0.0019, 0.0800),
        search_tilt_deg=5.0,
        touch_z_delta_xyz_m=(0.0, 0.0, -0.0041),
        touch_back_delta_xyz_m=(0.0020, 0.0, 0.0),
        touch_front_delta_xyz_m=(-0.0015, 0.0, -0.0010),
        touch_side_delta_xyz_m=(0.0, -0.0070, 0.0),
        assembly_delta_xyz_m=(0.0, 0.0, -0.0280),
        success_min_insertion_depth_m=0.020,
    ),
    "USB": TaskConfig(
        task_id="USB",
        # Bounding box von USB_Male.obj, auf Einfuehrachse Z abgebildet.
        peg_size_xyz_m=(0.015519, 0.008500, 0.050710),
        # USB_Male_Blender_collision_0.obj ist die in die Buchse eintauchende
        # Metallspitze: 12.000 x 4.480 mm. Version 1.4 verwendete irrtuemlich
        # das 15.519 x 8.500 mm grosse Gesamtgehaeuse als Einfuehrquerschnitt.
        peg_insertion_size_xy_m=(0.012000, 0.004480),
        # Summe der geschlossenen Visual-/Collisionmesh-Volumina bei der
        # MuJoCo-Standarddichte, gerundet.
        peg_mass_kg=0.00807,
        # Innenkanten der gelieferten USB_Female-Kollisionsmeshes:
        # rund 15.0 x 7.5 mm.
        hole_size_xy_m=(0.0150, 0.0075),
        socket_wall_height_m=0.030,
        socket_wall_thickness_m=0.008,
        socket_base_thickness_m=0.005,
        tcp_offset_above_peg_center_m=0.063,
        search_start_xyz_m=(0.0, -0.0012, 0.1200),
        search_tilt_deg=10.0,
        touch_z_delta_xyz_m=(-0.0026, 0.0, -0.0098),
        touch_back_delta_xyz_m=(0.0050, 0.0, 0.0),
        touch_front_delta_xyz_m=(-0.0050, 0.0, 0.0),
        touch_side_delta_xyz_m=(0.0, -0.0100, 0.0),
        assembly_delta_xyz_m=(0.0, 0.0, -0.0200),
        success_min_insertion_depth_m=0.014,
    ),
}


@dataclass(frozen=True)
class EnvironmentParameters:
    env_id: int
    task_id: str
    repetition: int
    socket_shift_x_m: float
    socket_shift_y_m: float
    stiffness_x_n_per_m: float
    stiffness_y_n_per_m: float
    stiffness_z_n_per_m: float
    damping_x_ns_per_m: float
    damping_y_ns_per_m: float
    damping_z_ns_per_m: float
    stiffness_rot_x_nm_per_rad: float
    stiffness_rot_y_nm_per_rad: float
    stiffness_rot_z_nm_per_rad: float
    damping_rot_x_nms_per_rad: float
    damping_rot_y_nms_per_rad: float
    damping_rot_z_nms_per_rad: float
    peg_socket_static_friction: float
    peg_socket_dynamic_friction: float
    gripper_peg_static_friction: float
    gripper_peg_dynamic_friction: float

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StudyConfig:
    tasks: List[str] = field(default_factory=lambda: ["KET12", "USB"])
    compliance_stiffness_values_n_per_m: List[float] = field(
        default_factory=lambda: [600.0, 1_200.0, 2_400.0, 4_800.0, 9_600.0]
    )
    socket_shift_y_values_mm: List[float] = field(
        default_factory=lambda: [-4.0, -2.0, 0.0, 2.0, 4.0]
    )
    repetitions: int = 2
    random_seed: int = 42
    environment_spacing_m: float = 0.35
    debug_environment_spacing_m: float = 1.40
    trace_stride_steps: int = 10


SOLVER = SolverConfig()
MUJOCO_COMPLIANCE = MujocoCompliance()
ISAAC_COMPLIANCE = IsaacCompliance()
CONTACT = ContactConfig()
STUDY = StudyConfig()

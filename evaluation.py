"""Einheitliche Ergebnis- und Fehlerklassifikation pro Umgebung."""

from dataclasses import dataclass, field

import numpy as np

from project_config import TASKS


RESULTS = (
    "SUCCESS",
    "FAILURE_1_CONTACT_LOSS_GRASP_APPROACH",
    "FAILURE_2_SOCKET_MISSED_DURING_SEARCH",
    "FAILURE_3_CONNECTOR_JAMMED",
    "FAILURE_4_CONTACT_LOSS_DURING_ASSEMBLY",
    "INVALID_SIMULATION_STATE",
)


@dataclass
class TrialMonitor:
    nominal_grip_error_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    grip_loss_threshold_m: float = 0.004
    maximum_lateral_deflection_m: float = 0.0
    maximum_axial_deflection_m: float = 0.0
    maximum_force_n: float = 0.0
    maximum_tilt_deg: float = 0.0
    maximum_grip_error_m: float = 0.0
    grasp_lost_before_assembly: bool = False
    grasp_lost_during_assembly: bool = False

    def __post_init__(self):
        self.nominal_grip_error_m = np.asarray(
            self.nominal_grip_error_m,
            dtype=float,
        ).copy()

    def observe(self, phase, deflection, total_force, tilt_deg, grip_error):
        deflection = np.asarray(deflection, dtype=float)
        total_force = np.asarray(total_force, dtype=float)
        grip_error = np.asarray(grip_error, dtype=float)
        self.maximum_lateral_deflection_m = max(
            self.maximum_lateral_deflection_m,
            float(np.linalg.norm(deflection[:2])),
        )
        self.maximum_axial_deflection_m = max(
            self.maximum_axial_deflection_m,
            abs(float(deflection[2])),
        )
        self.maximum_force_n = max(
            self.maximum_force_n,
            float(np.linalg.norm(total_force)),
        )
        self.maximum_tilt_deg = max(self.maximum_tilt_deg, float(tilt_deg))
        slip = float(
            np.linalg.norm(grip_error - self.nominal_grip_error_m)
        )
        self.maximum_grip_error_m = max(self.maximum_grip_error_m, slip)
        if slip > self.grip_loss_threshold_m:
            if phase in ("HOLD_ABOVE_SOCKET", "APPROACH"):
                self.grasp_lost_before_assembly = True
            else:
                self.grasp_lost_during_assembly = True


def upright_tilt_deg(quaternion_wxyz) -> float:
    q = np.asarray(quaternion_wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return float("nan")
    _, x, y, _ = q / norm
    cos_tilt = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_tilt)))


def evaluate_trial(
    task_id,
    peg_position,
    peg_orientation,
    socket_center_xy,
    monitor,
):
    task = TASKS[task_id]
    peg_position = np.asarray(peg_position, dtype=float)
    peg_orientation = np.asarray(peg_orientation, dtype=float)
    socket_center_xy = np.asarray(socket_center_xy, dtype=float)
    if not np.all(np.isfinite(peg_position)) or not np.all(
        np.isfinite(peg_orientation)
    ):
        return {
            "result": "INVALID_SIMULATION_STATE",
            "relative_x_m": float("nan"),
            "relative_y_m": float("nan"),
            "insertion_depth_m": float("nan"),
            "final_tilt_deg": float("nan"),
        }

    relative_xy = peg_position[:2] - socket_center_xy
    half_clearance = (
        np.asarray(task.hole_size_xy_m)
        - np.asarray(task.peg_insertion_size_xy_m)
    ) / 2.0 + task.evaluation_clearance_margin_m
    inside = bool(np.all(np.abs(relative_xy) <= half_clearance))
    peg_bottom_z = peg_position[2] - task.peg_size_xyz_m[2] / 2.0
    insertion_depth = max(0.0, task.socket_wall_height_m - peg_bottom_z)
    tilt = upright_tilt_deg(peg_orientation)

    if monitor.grasp_lost_before_assembly:
        result = "FAILURE_1_CONTACT_LOSS_GRASP_APPROACH"
    elif monitor.grasp_lost_during_assembly:
        result = "FAILURE_4_CONTACT_LOSS_DURING_ASSEMBLY"
    elif (
        inside
        and insertion_depth >= task.success_min_insertion_depth_m
        and tilt <= task.success_max_tilt_deg
    ):
        result = "SUCCESS"
    elif inside:
        result = "FAILURE_3_CONNECTOR_JAMMED"
    else:
        result = "FAILURE_2_SOCKET_MISSED_DURING_SEARCH"

    return {
        "result": result,
        "relative_x_m": float(relative_xy[0]),
        "relative_y_m": float(relative_xy[1]),
        "lateral_error_m": float(np.linalg.norm(relative_xy)),
        "insertion_depth_m": float(insertion_depth),
        "final_tilt_deg": float(tilt),
    }

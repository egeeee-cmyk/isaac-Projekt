"""Deterministische Parameterverteilung fuer 2 oder bis zu 100 Umgebungen."""

from itertools import product
from typing import Iterable, List, Optional

from project_config import (
    CONTACT,
    ISAAC_COMPLIANCE,
    STUDY,
    EnvironmentParameters,
)


def _base_combinations(
    tasks: Iterable[str],
    stiffness_values: Iterable[float],
    shifts_mm: Iterable[float],
    repetitions: int,
):
    return product(tasks, stiffness_values, shifts_mm, range(1, repetitions + 1))


def build_environment_parameters(
    num_envs: int,
    task_override: Optional[str] = None,
    validation_mode: bool = False,
) -> List[EnvironmentParameters]:
    """Erzeugt exakt ``num_envs`` Parametersaetze.

    Standardstudie:
        2 Aufgaben * 5 Steifigkeiten * 5 Versatzwerte * 2 Wiederholungen
        = 100 Umgebungen.

    Falls weniger Umgebungen angefordert werden, wird der Plan deterministisch
    abgeschnitten. Falls mehr angefordert werden, wird er zyklisch wiederholt
    und die Wiederholungsnummer fortgeschrieben.
    """

    if num_envs <= 0:
        raise ValueError("num_envs muss groesser als null sein.")
    if task_override is not None and task_override not in STUDY.tasks:
        raise ValueError(f"Unbekannte Aufgabe: {task_override}")

    if validation_mode:
        tasks = [task_override or "KET12"]
        stiffness_values = [ISAAC_COMPLIANCE.stiffness_x_n_per_m]
        shifts_mm = [0.0]
        repetitions = 1
    else:
        tasks = [task_override] if task_override else STUDY.tasks
        stiffness_values = STUDY.compliance_stiffness_values_n_per_m
        shifts_mm = STUDY.socket_shift_y_values_mm
        repetitions = STUDY.repetitions

    combinations = list(
        _base_combinations(tasks, stiffness_values, shifts_mm, repetitions)
    )
    if (
        num_envs == 2
        and task_override is None
        and not validation_mode
        and set(STUDY.tasks) == {"KET12", "USB"}
    ):
        # Praesentationsmodus: wirklich zwei unterschiedliche Aufgaben.
        combinations = [
            ("KET12", ISAAC_COMPLIANCE.stiffness_x_n_per_m, 0.0, 1),
            ("USB", ISAAC_COMPLIANCE.stiffness_x_n_per_m, 0.0, 1),
        ]
    if not combinations:
        raise ValueError("Der Versuchsplan ist leer.")

    parameters = []
    for env_id in range(num_envs):
        cycle = env_id // len(combinations)
        task_id, k_x, shift_mm, repetition = combinations[
            env_id % len(combinations)
        ]
        if task_id == "KET12":
            mu_static = CONTACT.gripper_peg_static_friction_ket12
            mu_dynamic = CONTACT.gripper_peg_dynamic_friction_ket12
        else:
            mu_static = CONTACT.gripper_peg_static_friction_usb
            mu_dynamic = CONTACT.gripper_peg_dynamic_friction_usb

        parameters.append(
            EnvironmentParameters(
                env_id=env_id,
                task_id=task_id,
                repetition=repetition + cycle * repetitions,
                socket_shift_x_m=0.0,
                socket_shift_y_m=float(shift_mm) / 1000.0,
                stiffness_x_n_per_m=float(k_x),
                stiffness_y_n_per_m=ISAAC_COMPLIANCE.stiffness_y_n_per_m,
                stiffness_z_n_per_m=ISAAC_COMPLIANCE.stiffness_z_n_per_m,
                damping_x_ns_per_m=ISAAC_COMPLIANCE.damping_x_ns_per_m,
                damping_y_ns_per_m=ISAAC_COMPLIANCE.damping_y_ns_per_m,
                damping_z_ns_per_m=ISAAC_COMPLIANCE.damping_z_ns_per_m,
                stiffness_rot_x_nm_per_rad=(
                    ISAAC_COMPLIANCE.stiffness_rot_x_nm_per_rad
                ),
                stiffness_rot_y_nm_per_rad=(
                    ISAAC_COMPLIANCE.stiffness_rot_y_nm_per_rad
                ),
                stiffness_rot_z_nm_per_rad=(
                    ISAAC_COMPLIANCE.stiffness_rot_z_nm_per_rad
                ),
                damping_rot_x_nms_per_rad=(
                    ISAAC_COMPLIANCE.damping_rot_x_nms_per_rad
                ),
                damping_rot_y_nms_per_rad=(
                    ISAAC_COMPLIANCE.damping_rot_y_nms_per_rad
                ),
                damping_rot_z_nms_per_rad=(
                    ISAAC_COMPLIANCE.damping_rot_z_nms_per_rad
                ),
                peg_socket_static_friction=(
                    CONTACT.peg_socket_static_friction
                ),
                peg_socket_dynamic_friction=(
                    CONTACT.peg_socket_dynamic_friction
                ),
                gripper_peg_static_friction=mu_static,
                gripper_peg_dynamic_friction=mu_dynamic,
            )
        )
    return parameters


def environment_origin(env_id: int, spacing_m: float = STUDY.environment_spacing_m):
    """Quadratisches Raster, passend fuer 2 und 100 sichtbare Umgebungen."""

    # Feste Breite verhindert, dass sich die Rasterkoordinaten beim Aufbau
    # spaeterer Umgebungen aendern. Env 0/1 liegen nebeneinander, 100
    # Umgebungen bilden ein 10x10-Raster.
    columns = 10
    row, column = divmod(env_id, columns)
    return (column * spacing_m, row * spacing_m, 0.0)

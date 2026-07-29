"""Feste, offene MuJoCo-Suchstrategie als reproduzierbare TCP-Bahn."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from project_config import TASKS


@dataclass(frozen=True)
class PoseTarget:
    position_xyz_m: np.ndarray
    tilt_x_deg: float
    phase: str


_TIMES = (0.0, 1.0, 2.5, 3.1, 3.7, 4.3, 5.0, 6.3, 7.0, 7.5)
_PHASES = (
    "HOLD_ABOVE_SOCKET",
    "APPROACH",
    "ANGLE_AND_TOUCH_Z",
    "TOUCH_BACK",
    "TOUCH_FRONT",
    "TOUCH_SIDE",
    "INSERT",
    "CENTER_AND_SEAT",
    "HOLD_INSERTED",
    "HOLD_INSERTED",
)


def task_waypoints(task_id: str) -> List[Tuple[float, np.ndarray, float, str]]:
    task = TASKS[task_id]
    p0 = np.asarray(task.search_start_xyz_m, dtype=float)
    # Vor der Suchphase wird nur oberhalb des Suchstarts gehalten.
    above = p0 + np.array([0.0, 0.0, 0.035])
    touch_z = p0 + np.asarray(task.touch_z_delta_xyz_m)
    touch_back = touch_z + np.asarray(task.touch_back_delta_xyz_m)
    touch_front = touch_back + np.asarray(task.touch_front_delta_xyz_m)
    touch_side = touch_front + np.asarray(task.touch_side_delta_xyz_m)
    assembly = touch_side + np.asarray(task.assembly_delta_xyz_m)
    # Die vereinfachte Isaac-Szene besitzt eine ebene Arbeitsplatte bei z=0.
    # Die uebernommene KET12-Bahn wuerde den Peg 3,1 mm in diese Platte
    # fahren. Ein Millimeter Bodenfreiheit verhindert diesen kuenstlichen
    # Kontakt, ohne die geforderte Einsetztiefe zu unterschreiten.
    minimum_tcp_z = (
        task.tcp_offset_above_peg_center_m
        + task.peg_size_xyz_m[2] / 2.0
        + 0.001
    )
    assembly[2] = max(assembly[2], minimum_tcp_z)
    # Die MuJoCo-Suchbewegung endet seitlich an der Buchsenwand. Fuer den
    # eigentlichen Montagezustand wird der TCP anschliessend wieder auf die
    # nominelle Buchsenmitte gefuehrt und dort gehalten. Ohne diesen Schritt
    # bewertete die alte Version beide Nullversatz-Referenzen zwangslaeufig als
    # "socket missed" (KET12: -5.1 mm, USB: -11.2 mm in Y).
    centered_seat = assembly.copy()
    centered_seat[:2] = 0.0
    preinsert = assembly
    phases = _PHASES
    if task_id == "USB":
        # Beim Zentrieren darf nicht der geneigte TCP auf XY=(0, 0) gesetzt
        # werden. Stattdessen wird der aufgabenspezifische, aus der
        # MuJoCo-Griffpose abgeleitete TCP-Stecker-Abstand beruecksichtigt.
        #
        # Die korrigierte Bahn hebt zuerst mit geometrisch berechneter
        # Freigabe an, zentriert das *Steckerzentrum* bei unveraenderter
        # Neigung, richtet oberhalb der Buchse auf und setzt erst dann
        # ausschliesslich vertikal ein.
        tilt_rad = np.deg2rad(task.search_tilt_deg)
        half_y = task.peg_insertion_size_xy_m[1] / 2.0
        half_z = task.peg_size_xyz_m[2] / 2.0
        clearance = 0.003
        required_clear_tcp_z = (
            task.socket_wall_height_m
            + clearance
            + (task.tcp_offset_above_peg_center_m + half_z)
            * np.cos(tilt_rad)
            + half_y * abs(np.sin(tilt_rad))
        )
        raised_at_side = touch_side.copy()
        # Ist die MuJoCo-Suchpose bereits hoeher als die berechnete
        # Mindestfreigabe, darf dieser Schritt den Greifer nicht absenken.
        raised_at_side[2] = max(touch_side[2], required_clear_tcp_z)

        centered_tilted = raised_at_side.copy()
        centered_tilted[0] = 0.0
        centered_tilted[1] = (
            -task.tcp_offset_above_peg_center_m * np.sin(tilt_rad)
        )

        preinsert = centered_tilted.copy()
        preinsert[:2] = 0.0
        preinsert[2] = (
            task.socket_wall_height_m
            + clearance
            + task.tcp_offset_above_peg_center_m
            + half_z
        )
        centered_seat = preinsert + np.asarray(task.assembly_delta_xyz_m)
        centered_seat[:2] = 0.0
        centered_seat[2] = max(centered_seat[2], minimum_tcp_z)
        times = (
            0.0,
            1.0,
            2.5,
            3.1,
            3.7,
            4.3,
            5.0,
            5.5,
            5.9,
            6.3,
            7.0,
            7.5,
        )
        phases = (
            "HOLD_ABOVE_SOCKET",
            "APPROACH",
            "ANGLE_AND_TOUCH_Z",
            "TOUCH_BACK",
            "TOUCH_FRONT",
            "TOUCH_SIDE",
            "TOUCH_SIDE",
            "RAISE_CLEAR_OF_SOCKET",
            "CENTER_PEG_WHILE_TILTED",
            "STRAIGHTEN_ABOVE_SOCKET",
            "CENTER_AND_SEAT",
            "HOLD_INSERTED",
        )
        positions = (
            above,
            above,
            p0,
            touch_z,
            touch_back,
            touch_front,
            touch_side,
            raised_at_side,
            centered_tilted,
            preinsert,
            centered_seat,
            centered_seat,
        )
        tilts = (
            0.0,
            0.0,
            task.search_tilt_deg,
            task.search_tilt_deg,
            task.search_tilt_deg,
            task.search_tilt_deg,
            task.search_tilt_deg,
            task.search_tilt_deg,
            task.search_tilt_deg,
            0.0,
            0.0,
            0.0,
        )
        return list(zip(times, positions, tilts, phases))

    positions = (
        above,
        above,
        p0,
        touch_z,
        touch_back,
        touch_front,
        touch_side,
        preinsert,
        centered_seat,
        centered_seat,
    )
    tilts = (
        0.0,
        0.0,
        task.search_tilt_deg,
        task.search_tilt_deg,
        task.search_tilt_deg,
        task.search_tilt_deg,
        task.search_tilt_deg,
        task.search_tilt_deg,
        0.0,
        0.0,
    )
    return list(zip(_TIMES, positions, tilts, phases))


def target_pose(task_id: str, time_s: float) -> PoseTarget:
    waypoints = task_waypoints(task_id)
    if time_s <= waypoints[0][0]:
        _, pos, tilt, phase = waypoints[0]
        return PoseTarget(pos.copy(), tilt, phase)
    if time_s >= waypoints[-1][0]:
        _, pos, tilt, phase = waypoints[-1]
        return PoseTarget(pos.copy(), tilt, phase)

    for left, right in zip(waypoints[:-1], waypoints[1:]):
        t0, p0, a0, _ = left
        t1, p1, a1, phase = right
        if t0 <= time_s < t1:
            alpha = (time_s - t0) / (t1 - t0)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            position = p0 * (1.0 - smooth) + p1 * smooth
            tilt = a0 * (1.0 - smooth) + a1 * smooth
            return PoseTarget(position, float(tilt), phase)
    raise RuntimeError("Ungueltige Zeitinterpolation.")


def quaternion_x_deg(angle_deg: float) -> np.ndarray:
    half = np.deg2rad(angle_deg) / 2.0
    return np.array([np.cos(half), np.sin(half), 0.0, 0.0], dtype=float)

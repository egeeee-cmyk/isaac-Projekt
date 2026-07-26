

import argparse
from pathlib import Path
import sys
import traceback

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ASSET_DIR = PROJECT_DIR / "assets" / "mujoco"
REFERENCE_DIR = PROJECT_DIR / "reference_mujoco"
RESULTS_DIR = PROJECT_DIR / "results"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--debug-two-envs", action="store_true")
    parser.add_argument("--single-validation", action="store_true")
    parser.add_argument("--task", choices=("KET12", "USB"), default=None)
    parser.add_argument("--physics-dt", type=float, default=None)
    parser.add_argument(
        "--show-ur5e",
        action="store_true",
        help="Laedt die UR5e-/Finray-CAD-Geometrie im sichtbaren Debuglauf.",
    )
    parser.add_argument(
        "--trace-all",
        action="store_true",
        help="Zeitverlauf aller statt nur der ersten zwei Umgebungen.",
    )
    parser.add_argument(
        "--camera-view",
        choices=("overview", "ket12", "usb"),
        default="overview",
        help=(
            "Praesentationskamera: Gesamtansicht oder Nahansicht einer "
            "Montageaufgabe."
        ),
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Isaac/Kit-Zusatzargumente werden ignoriert: {unknown}")
    if args.debug_two_envs:
        args.num_envs = 2
        args.headless = False
        args.show_ur5e = True
    if args.single_validation:
        args.num_envs = 1
        args.headless = True
    if args.num_envs <= 0:
        parser.error("--num-envs muss groesser als null sein.")
    return args


ARGS = parse_arguments()

from isaac_imports import (
    import_core_api,
    import_set_camera_view,
    import_simulation_app,
)

SimulationApp = import_simulation_app()
simulation_app = SimulationApp({"headless": ARGS.headless})
World, DynamicCuboid, FixedCuboid = import_core_api()
set_camera_view = import_set_camera_view()

import omni.usd
from omni.kit.viewport.utility import (
    get_active_viewport,
    get_active_viewport_camera_path,
)
from pxr import Sdf

from aggregation import aggregate_results, write_manifest, write_rows
from evaluation import TrialMonitor, evaluate_trial, upright_tilt_deg
from motion import quaternion_x_deg, target_pose
from parameter_sweep import build_environment_parameters, environment_origin
from project_config import SOLVER, STUDY, TASKS
from robot_visual import (
    author_ur5e_finray_visual,
    presentation_base_translation,
    presentation_joint_positions,
    presentation_joint_positions_at,
    update_authored_joint_positions,
)
from scene import (
    RENDER_CAMERA_PATH,
    create_environment,
    create_render_environment,
    reset_environment,
    set_target,
)


def _get_pose(obj):
    position, orientation = obj.get_world_pose()
    return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)


def _configure_camera(environments):
    if ARGS.headless:
        return
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Kein aktiver Isaac-Sim-Viewport gefunden.")
    try:
        viewport.set_active_camera(RENDER_CAMERA_PATH)
    except TypeError:
        viewport.set_active_camera(Sdf.Path(RENDER_CAMERA_PATH))

    if ARGS.camera_view == "overview":
        origins = np.asarray([env.origin for env in environments], dtype=float)
        target = np.mean(origins, axis=0) + np.array([0.0, 0.0, 0.38])
        eye = target + np.array([1.55, 2.45, 1.25])
    else:
        requested_task = ARGS.camera_view.upper()
        selected = next(
            (
                env
                for env in environments
                if env.parameters.task_id == requested_task
            ),
            environments[0],
        )
        target = selected.origin + np.array([0.0, 0.0, 0.055])
        eye = target + np.array([0.20, 0.34, 0.18])
    set_camera_view(
        eye=eye,
        target=target,
        camera_prim_path=RENDER_CAMERA_PATH,
    )
    active_camera = get_active_viewport_camera_path()
    if active_camera is None or str(active_camera) != RENDER_CAMERA_PATH:
        raise RuntimeError(
            "Projektkamera wurde nicht als aktive Viewport-Kamera übernommen: "
            f"{active_camera}"
        )
    print(
        f"Viewport bereit, Kamera={ARGS.camera_view}: {active_camera}",
        flush=True,
    )


def _apply_solver_iterations(stage, environments):
    try:
        from pxr import PhysxSchema

        for env in environments:
            for prim_path in (env.peg.prim_path, env.gripper.frame.prim_path):
                prim = stage.GetPrimAtPath(prim_path)
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                api.CreateSolverPositionIterationCountAttr(
                    SOLVER.position_iterations
                )
                api.CreateSolverVelocityIterationCountAttr(
                    SOLVER.velocity_iterations
                )
    except Exception as exc:
        print(f"Hinweis: Solver-Iterationsattribute nicht gesetzt: {exc}")


def run():
    physics_dt = ARGS.physics_dt or (
        SOLVER.mujoco_reference_dt_s
        if ARGS.single_validation
        else SOLVER.study_physics_dt_s
    )
    parameters = build_environment_parameters(
        ARGS.num_envs,
        task_override=ARGS.task,
        validation_mode=ARGS.single_validation,
    )
    print(
        f"Starte {len(parameters)} Umgebungen, dt={physics_dt:g} s, "
        f"headless={ARGS.headless}.",
        flush=True,
    )
    print("[Setup 1/6] Physikwelt erzeugen ...", flush=True)
    world = World(
        physics_dt=physics_dt,
        rendering_dt=SOLVER.rendering_dt_s,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()
    if not ARGS.headless:
        create_render_environment(stage)
        simulation_app.update()

    print("[Setup 2/6] Montageumgebungen erzeugen ...", flush=True)
    environments = []
    debug_spacing = (
        STUDY.debug_environment_spacing_m
        if ARGS.debug_two_envs
        else STUDY.environment_spacing_m
    )
    for parameter in parameters:
        runtime = create_environment(
            world,
            stage,
            DynamicCuboid,
            FixedCuboid,
            parameter,
            environment_origin(parameter.env_id, spacing_m=debug_spacing),
            ASSET_DIR,
            show_finray_visual=(not ARGS.headless and parameter.env_id < 2),
        )
        environments.append(runtime)
        if not ARGS.headless:
            simulation_app.update()
        print(
            f"  Umgebung {parameter.env_id + 1}/{len(parameters)}: "
            f"{parameter.task_id} "
            f"(Einfuehrquerschnitt "
            f"{1000.0 * TASKS[parameter.task_id].peg_insertion_size_xy_m[0]:.2f}"
            " x "
            f"{1000.0 * TASKS[parameter.task_id].peg_insertion_size_xy_m[1]:.2f}"
            " mm)",
            flush=True,
        )

    print("[Setup 3/6] Physik initialisieren ...", flush=True)
    _apply_solver_iterations(stage, environments)
    world.reset()
    for env in environments:
        reset_environment(env)
    if not ARGS.headless:
        _configure_camera(environments)
        for _ in range(12):
            world.step(render=True)

    robot_placements = {}
    if ARGS.show_ur5e and not ARGS.headless:
        print(
            "[Setup 4/6] UR5e-CAD laden; das kann kurz dauern ...",
            flush=True,
        )
        for index, env in enumerate(environments[:2], start=1):
            initial_target = target_pose(env.parameters.task_id, 0.0)
            desired_tcp = env.origin + initial_target.position_xyz_m
            joint_positions = presentation_joint_positions(
                env.parameters.task_id
            )
            placement = author_ur5e_finray_visual(
                stage,
                f"{env.root_path}/UR5eCAD",
                REFERENCE_DIR / "robot_15.xml",
                ASSET_DIR,
                presentation_base_translation(
                    env.parameters.task_id,
                    env.origin,
                ),
                joint_positions=joint_positions,
                progress_callback=simulation_app.update,
            )
            robot_placements[env.parameters.env_id] = placement
            print(
                f"  UR5e {index}/{min(2, len(environments))} geladen.",
                flush=True,
            )
            tcp_error = np.linalg.norm(
                placement["tcp_world"] - desired_tcp
            )
            print(
                f"    CAD-TCP-Ausrichtfehler: {1000.0 * tcp_error:.6f} mm",
                flush=True,
            )
        for _ in range(8):
            world.step(render=True)
    else:
        print("[Setup 4/6] UR5e-CAD übersprungen.", flush=True)

    print("[Setup 5/6] Messung vorbereiten ...", flush=True)
    monitors = [
        TrialMonitor(
            nominal_grip_error_m=env.gripper.grip_error(_get_pose(env.peg)[0]),
            grip_loss_threshold_m=TASKS[
                env.parameters.task_id
            ].grip_loss_threshold_m,
        )
        for env in environments
    ]
    trace_rows = []
    previous_reference_positions = [
        env.gripper.get_reference_position().copy() for env in environments
    ]
    steps = int(round(SOLVER.total_time_s / physics_dt))
    trace_limit = len(environments) if ARGS.trace_all else min(2, len(environments))

    print(
        f"[Setup 6/6] Szene sichtbar; Simulation startet ({steps} Schritte).",
        flush=True,
    )
    render_stride = max(
        1,
        int(round(SOLVER.rendering_dt_s / physics_dt)),
    )
    for step in range(steps + 1):
        time_s = min(step * physics_dt, SOLVER.total_time_s)
        phases = [set_target(env, time_s) for env in environments]
        render_this_step = (
            not ARGS.headless
            and (step % render_stride == 0 or step == steps)
        )
        if render_this_step:
            for env in environments:
                placement = robot_placements.get(env.parameters.env_id)
                if placement is not None:
                    update_authored_joint_positions(
                        placement,
                        presentation_joint_positions_at(
                            env.parameters.task_id,
                            time_s,
                        ),
                    )
        world.step(render=render_this_step)

        for index, (env, phase, monitor) in enumerate(
            zip(environments, phases, monitors)
        ):
            peg_position, peg_orientation = _get_pose(env.peg)
            frame_position = env.gripper.get_grasp_frame_position()
            reference_position = env.gripper.get_reference_position()
            reference_velocity = (
                reference_position - previous_reference_positions[index]
            ) / physics_dt
            previous_reference_positions[index] = reference_position.copy()
            state = env.compliance_joint.measure(
                reference_position,
                frame_position,
                reference_velocity,
                env.gripper.frame.get_linear_velocity(),
            )
            tilt = upright_tilt_deg(peg_orientation)
            grip_error = env.gripper.grip_error(peg_position)
            monitor.observe(
                phase,
                state["deflection_m"],
                state["total_force_n"],
                tilt,
                grip_error,
            )
            if (
                index < trace_limit
                and step % STUDY.trace_stride_steps == 0
            ):
                trace_rows.append(
                    {
                        "env_id": env.parameters.env_id,
                        "task_id": env.parameters.task_id,
                        "step": step,
                        "time_s": time_s,
                        "phase": phase,
                        "peg_x_m": peg_position[0] - env.origin[0],
                        "peg_y_m": peg_position[1] - env.origin[1],
                        "peg_z_m": peg_position[2],
                        "deflection_x_m": state["deflection_m"][0],
                        "deflection_y_m": state["deflection_m"][1],
                        "deflection_z_m": state["deflection_m"][2],
                        "force_x_n": state["total_force_n"][0],
                        "force_y_n": state["total_force_n"][1],
                        "force_z_n": state["total_force_n"][2],
                        "tilt_deg": tilt,
                    }
                )

    rows = []
    for env, monitor in zip(environments, monitors):
        peg_position, peg_orientation = _get_pose(env.peg)
        result = evaluate_trial(
            env.parameters.task_id,
            peg_position,
            peg_orientation,
            env.socket_center_xy,
            monitor,
        )
        row = env.parameters.as_dict()
        row.update(result)
        row.update(
            {
                "socket_shift_x_mm": 1000.0
                * env.parameters.socket_shift_x_m,
                "socket_shift_y_mm": 1000.0
                * env.parameters.socket_shift_y_m,
                "maximum_lateral_deflection_m": (
                    monitor.maximum_lateral_deflection_m
                ),
                "maximum_axial_deflection_m": (
                    monitor.maximum_axial_deflection_m
                ),
                "maximum_force_n": monitor.maximum_force_n,
                "maximum_tilt_deg": monitor.maximum_tilt_deg,
                "maximum_grip_error_m": monitor.maximum_grip_error_m,
                "grasp_lost_before_assembly": (
                    monitor.grasp_lost_before_assembly
                ),
                "grasp_lost_during_assembly": (
                    monitor.grasp_lost_during_assembly
                ),
                "final_peg_x_local_m": peg_position[0] - env.origin[0],
                "final_peg_y_local_m": peg_position[1] - env.origin[1],
                "final_peg_z_m": peg_position[2],
                "physics_dt_s": physics_dt,
            }
        )
        rows.append(row)

    RESULTS_DIR.mkdir(exist_ok=True)
    write_rows(RESULTS_DIR / "environment_results.csv", rows)
    write_rows(RESULTS_DIR / "aggregate_results.csv", aggregate_results(rows))
    if trace_rows:
        write_rows(RESULTS_DIR / "debug_trace.csv", trace_rows)
    write_manifest(RESULTS_DIR / "run_manifest.json", ARGS, parameters)
    successes = sum(row["result"] == "SUCCESS" for row in rows)
    for row in rows:
        print(
            "  "
            f"Env {row['env_id']:03d} {row['task_id']}: "
            f"{row['result']}, "
            f"Fehler={1000.0 * row.get('lateral_error_m', float('nan')):.3f} mm, "
            f"Einsetztiefe={1000.0 * row['insertion_depth_m']:.3f} mm, "
            f"Neigung={row['final_tilt_deg']:.3f} deg",
            flush=True,
        )
    print(
        f"Abgeschlossen: {successes}/{len(rows)} erfolgreich. "
        f"Ergebnisse: {RESULTS_DIR}"
    )
    if ARGS.debug_two_envs:
        print(
            "Debugansicht bleibt offen. "
            "Zum Beenden das Isaac-Sim-Fenster schliessen."
        )
        world.pause()
        while simulation_app.is_running():
            simulation_app.update()


if __name__ == "__main__":
    try:
        run()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

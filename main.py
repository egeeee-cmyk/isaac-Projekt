"""Parallele MuJoCo-basierte Montageversuche in NVIDIA Isaac Sim
"""

import argparse
from pathlib import Path
import sys
import traceback

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ASSET_DIR = PROJECT_DIR / "assets" / "mujoco"
REFERENCE_DIR = PROJECT_DIR / "reference_mujoco"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument(
        "--headless-study-100",
        action="store_true",
        help=(
            "Erzwingt den freigegebenen Studienmodus: headless, 100 "
            "Umgebungen, 50 KET12 und 50 USB."
        ),
    )
    parser.add_argument(
        "--visual-demo",
        action="store_true",
        help=(
            "Startet zwei sichtbare Umgebungen mit UR5e-CAD und exportiert "
            "Start-/Endbilder für Übersicht, KET12 und USB."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Ergebnisordner, absolut oder relativ zum Projekt. Standard im "
            "100er-Modus: results/headless_100."
        ),
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help=(
            "Ueberschreibt ausschliesslich bekannte Ergebnisdateien im "
            "gewaehlten Ausgabeordner."
        ),
    )
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
    parser.add_argument(
        "--export-renderings",
        action="store_true",
        help=(
            "Exportiert im sichtbaren Modus reproduzierbare PNG-Ansichten "
            "unterhalb des Ergebnisordners."
        ),
    )
    parser.add_argument(
        "--render-views",
        nargs="+",
        choices=("overview", "ket12", "usb"),
        default=("overview", "ket12", "usb"),
        help="Zu exportierende Präsentationsansichten.",
    )
    parser.add_argument(
        "--render-width",
        type=int,
        default=1920,
        help="Breite der PNG-Renderings; Standard 1920.",
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=1080,
        help="Höhe der PNG-Renderings; Standard 1080.",
    )
    parser.add_argument(
        "--renderer",
        choices=("RaytracedLighting", "PathTracing"),
        default="RaytracedLighting",
        help="Isaac-Renderer für Viewport und PNG-Export.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Isaac/Kit-Zusatzargumente werden ignoriert: {unknown}")
    if args.visual_demo and (
        args.headless or args.headless_study_100 or args.single_validation
    ):
        parser.error(
            "--visual-demo ist nicht mit einem Headless-/Validierungsmodus "
            "kombinierbar."
        )
    if args.visual_demo and args.task is not None:
        parser.error(
            "--visual-demo enthält fest KET12 und USB und darf nicht mit "
            "--task kombiniert werden."
        )
    if args.visual_demo:
        args.debug_two_envs = True
        args.export_renderings = True
        if args.output_dir is None:
            args.output_dir = "results/visual_demo"
    if args.headless_study_100:
        if args.task is not None:
            parser.error(
                "--headless-study-100 darf nicht mit --task kombiniert werden."
            )
        if args.debug_two_envs or args.single_validation:
            parser.error(
                "--headless-study-100 ist nicht mit Debug-/Validierungsmodus "
                "kombinierbar."
            )
        args.headless = True
        args.num_envs = 100
        args.trace_all = False
    if args.debug_two_envs:
        args.num_envs = 2
        args.headless = False
        args.show_ur5e = True
    if args.single_validation:
        args.num_envs = 1
        args.headless = True
    if args.export_renderings and args.headless:
        parser.error("--export-renderings benötigt einen sichtbaren Isaac-Modus.")
    if (
        args.render_width < 640
        or args.render_height < 360
        or args.render_width > 3840
        or args.render_height > 2160
        or abs(args.render_width / args.render_height - 16.0 / 9.0) > 0.02
    ):
        parser.error(
            "--render-width/--render-height müssen 16:9 und zwischen "
            "640x360 und 3840x2160 liegen."
        )
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
launch_config = {"headless": ARGS.headless}
if not ARGS.headless:
    launch_config.update(
        {
            "width": ARGS.render_width,
            "height": ARGS.render_height,
            "window_width": min(ARGS.render_width, 1600),
            "window_height": min(ARGS.render_height, 1000),
            "renderer": ARGS.renderer,
        }
    )
simulation_app = SimulationApp(launch_config)
World, DynamicCuboid, FixedCuboid = import_core_api()
set_camera_view = import_set_camera_view() if not ARGS.headless else None

import omni.usd

if not ARGS.headless:
    from pxr import Sdf
    from omni.kit.viewport.utility import (
        get_active_viewport,
        get_active_viewport_camera_path,
    )
else:
    Sdf = None
    get_active_viewport = None
    get_active_viewport_camera_path = None

from aggregation import aggregate_results, write_rows
from evaluation import TrialMonitor, evaluate_trial, upright_tilt_deg
from motion import quaternion_x_deg, target_pose
from parameter_sweep import build_environment_parameters, environment_origin
from project_config import SOLVER, STUDY, TASKS
from project_version import PROJECT_VERSION
from rendering import (
    camera_pose,
    capture_presentation_checkpoint,
    configure_viewport_resolution,
    write_render_manifest,
)
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
from study_reporting import (
    parameter_plan_rows,
    prepare_results_directory,
    resolve_results_dir,
    utc_now_iso,
    validate_headless_100_plan,
    verify_physics_baseline,
    write_automatic_evaluation,
    write_json,
    write_run_manifest,
)


RUN_CONTEXT = {}


def _get_pose(obj):
    position, orientation = obj.get_world_pose()
    return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)


def _configure_camera(environments, view_name=None):
    if ARGS.headless:
        return
    view_name = view_name or ARGS.camera_view
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Kein aktiver Isaac-Sim-Viewport gefunden.")
    try:
        viewport.set_active_camera(RENDER_CAMERA_PATH)
    except TypeError:
        viewport.set_active_camera(Sdf.Path(RENDER_CAMERA_PATH))

    configure_viewport_resolution(
        viewport,
        ARGS.render_width,
        ARGS.render_height,
    )
    eye, target = camera_pose(view_name, environments)
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
        f"Viewport bereit, Kamera={view_name}, "
        f"Auflösung={viewport.resolution}: {active_camera}",
        flush=True,
    )
    return viewport


def _capture_renderings(environments, results_dir, checkpoint):
    if not ARGS.export_renderings:
        return []
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Kein aktiver Viewport für PNG-Export gefunden.")
    return capture_presentation_checkpoint(
        simulation_app=simulation_app,
        viewport=viewport,
        environments=environments,
        configure_camera=lambda view: _configure_camera(environments, view),
        output_dir=Path(results_dir) / "renderings",
        checkpoint=checkpoint,
        views=ARGS.render_views,
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
    started_at_utc = utc_now_iso()
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
    results_dir = resolve_results_dir(PROJECT_DIR, ARGS)
    prepare_results_directory(
        results_dir,
        overwrite=ARGS.overwrite_results,
    )
    RUN_CONTEXT.update(
        {
            "started_at_utc": started_at_utc,
            "physics_dt_s": physics_dt,
            "parameters": parameters,
            "results_dir": results_dir,
        }
    )

    baseline_check = verify_physics_baseline(PROJECT_DIR)
    write_json(results_dir / "physics_baseline_check.json", baseline_check)
    if not baseline_check["passed"]:
        raise RuntimeError(
            "Physikalisches v1.8-Grundmodell stimmt nicht mit der "
            "gesicherten Baseline ueberein."
        )

    if ARGS.headless_study_100:
        plan_check = validate_headless_100_plan(parameters)
        write_json(results_dir / "parameter_plan_check.json", plan_check)
        if not plan_check["passed"]:
            raise RuntimeError(
                "Ungueltiger Headless-100er-Plan: "
                + " ".join(plan_check["issues"])
            )
        write_rows(
            results_dir / "parameter_plan_100.csv",
            parameter_plan_rows(parameters),
        )
    write_run_manifest(
        results_dir / "run_manifest.json",
        ARGS,
        parameters,
        physics_dt,
        status="RUNNING",
        started_at_utc=started_at_utc,
    )
    print(
        f"Starte {len(parameters)} Umgebungen, dt={physics_dt:g} s, "
        f"headless={ARGS.headless}, Ergebnisse={results_dir}.",
        flush=True,
    )
    print("[Setup 1/6] Physikwelt erzeugen ...", flush=True)
    world = World(
        physics_dt=physics_dt,
        rendering_dt=SOLVER.rendering_dt_s,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()
    render_records = []
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
            show_original_visuals=(not ARGS.headless and parameter.env_id < 2),
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

    if ARGS.export_renderings:
        print(
            "[Rendering] Startzustand aus Übersicht, KET12 und USB ...",
            flush=True,
        )
        world.pause()
        render_records.extend(
            _capture_renderings(environments, results_dir, "start")
        )
        _configure_camera(environments)
        world.play()

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

    if ARGS.export_renderings:
        print(
            "[Rendering] Endzustand aus Übersicht, KET12 und USB ...",
            flush=True,
        )
        world.pause()
        render_records.extend(
            _capture_renderings(environments, results_dir, "final")
        )
        write_render_manifest(
            results_dir / "render_manifest.json",
            render_records,
            renderer=ARGS.renderer,
            width=ARGS.render_width,
            height=ARGS.render_height,
            project_version=PROJECT_VERSION,
        )
        _configure_camera(environments)

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

    write_rows(results_dir / "environment_results.csv", rows)
    write_rows(
        results_dir / "aggregate_results.csv",
        aggregate_results(rows),
    )
    if trace_rows:
        write_rows(results_dir / "debug_trace.csv", trace_rows)
    expected_count = 100 if ARGS.headless_study_100 else len(parameters)
    automatic_report = write_automatic_evaluation(
        results_dir,
        rows,
        expected_count=expected_count,
    )
    write_run_manifest(
        results_dir / "run_manifest.json",
        ARGS,
        parameters,
        physics_dt,
        status="COMPLETED",
        started_at_utc=started_at_utc,
        finished_at_utc=utc_now_iso(),
    )
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
        f"Qualitaetsgate="
        f"{'BESTANDEN' if automatic_report['quality_gate_passed'] else 'NICHT BESTANDEN'}. "
        f"Ergebnisse: {results_dir}"
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
        error_trace = traceback.format_exc()
        traceback.print_exc()
        if RUN_CONTEXT:
            try:
                write_run_manifest(
                    RUN_CONTEXT["results_dir"] / "run_manifest.json",
                    ARGS,
                    RUN_CONTEXT["parameters"],
                    RUN_CONTEXT["physics_dt_s"],
                    status="FAILED",
                    started_at_utc=RUN_CONTEXT["started_at_utc"],
                    finished_at_utc=utc_now_iso(),
                    error=error_trace,
                )
            except Exception:
                traceback.print_exc()
        raise
    finally:
        simulation_app.close()

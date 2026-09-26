from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


PRESENTATION_VIEWS = ("overview", "ket12", "usb")
PRESENTATION_CHECKPOINTS = ("start", "final")


def validate_render_resolution(width, height):
    """Validiert eine praxisgerechte 16:9-Ausgabeauflösung."""

    width = int(width)
    height = int(height)
    if width < 640 or height < 360:
        raise ValueError("Rendering-Auflösung muss mindestens 640 x 360 sein.")
    if width > 3840 or height > 2160:
        raise ValueError(
            "Rendering-Auflösung ist auf höchstens 3840 x 2160 begrenzt."
        )
    if abs(width / height - 16.0 / 9.0) > 0.02:
        raise ValueError(
            "Für reproduzierbare Präsentationsbilder ist 16:9 erforderlich."
        )
    return width, height


def camera_pose(view_name, environments):
    """Berechnet feste Kamera- und Zielpositionen für eine Projektansicht."""

    view_name = str(view_name).lower()
    if view_name not in PRESENTATION_VIEWS:
        raise ValueError(f"Unbekannte Kameraansicht: {view_name}")
    environments = list(environments)
    if not environments:
        raise ValueError("Mindestens eine Umgebung wird für die Kamera benötigt.")

    if view_name == "overview":
        origins = np.asarray(
            [np.asarray(env.origin, dtype=float) for env in environments],
            dtype=float,
        )
        target = np.mean(origins, axis=0) + np.array([0.0, 0.0, 0.38])
        horizontal_extent = max(
            float(np.ptp(origins[:, 0])),
            float(np.ptp(origins[:, 1])),
            0.6,
        )
        # Die Zwei-Umgebungs-Ansicht bleibt nah genug; bei 100 Umgebungen
        # wächst der Abstand automatisch mit dem 10x10-Raster.
        eye = target + np.array(
            [
                max(1.55, 0.90 * horizontal_extent),
                max(2.45, 1.50 * horizontal_extent),
                max(1.25, 0.75 * horizontal_extent),
            ]
        )
        return eye, target

    requested_task = view_name.upper()
    selected = next(
        (
            env
            for env in environments
            if env.parameters.task_id == requested_task
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            f"Keine {requested_task}-Umgebung für die Kamera vorhanden."
        )
    target = np.asarray(selected.origin, dtype=float) + np.array(
        [0.0, 0.0, 0.055]
    )
    eye = target + np.array([0.20, 0.34, 0.18])
    return eye, target


def render_filename(checkpoint, view_name):
    checkpoint = str(checkpoint).lower()
    view_name = str(view_name).lower()
    if checkpoint not in PRESENTATION_CHECKPOINTS:
        raise ValueError(f"Unbekannter Rendering-Zeitpunkt: {checkpoint}")
    if view_name not in PRESENTATION_VIEWS:
        raise ValueError(f"Unbekannte Kameraansicht: {view_name}")
    return f"{checkpoint}_{view_name}.png"


def configure_viewport_resolution(viewport, width, height):
    """Setzt die tatsächliche Rendertextur unabhängig von der Fenstergröße."""

    width, height = validate_render_resolution(width, height)
    viewport.fill_frame = False
    viewport.resolution = (width, height)
    actual = tuple(int(value) for value in viewport.resolution)
    if actual != (width, height):
        raise RuntimeError(
            "Viewport-Auflösung wurde nicht übernommen: "
            f"erwartet {(width, height)}, erhalten {actual}."
        )
    return actual


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wait_for_capture(
    capture_helper,
    simulation_app,
    max_updates=600,
    completion_frames=30,
):
    """Pumpt Kit-Updates, bis der asynchrone Viewport-Export abgeschlossen ist.

    ``capture_viewport_to_file`` schreibt die Datei asynchron. Isaac Sim 5.x
    unterstützt ``completion_frames``; der Fallback hält die Funktion auch mit
    älteren Kit-Versionen kompatibel.
    """

    async def _wait_for_result():
        try:
            return await capture_helper.wait_for_result(
                completion_frames=int(completion_frames)
            )
        except TypeError:
            return await capture_helper.wait_for_result()

    wait_task = asyncio.ensure_future(_wait_for_result())
    for _ in range(int(max_updates)):
        simulation_app.update()
        if wait_task.done():
            break
    if not wait_task.done():
        wait_task.cancel()
        raise TimeoutError("Viewport-Aufnahme wurde nicht rechtzeitig beendet.")
    return wait_task.result()


def _wait_for_file(path, simulation_app, max_updates=120):
    """Wartet nach dem Kit-Future zusätzlich auf den tatsächlichen Dateieintrag."""

    path = Path(path)
    for _ in range(int(max_updates)):
        simulation_app.update()
        if path.is_file() and path.stat().st_size > 0:
            return True
    return path.is_file() and path.stat().st_size > 0


def capture_viewport_png(
    simulation_app,
    viewport,
    output_path,
    warmup_updates=12,
):
    """Speichert den aktuellen LDR-Viewport über die offizielle Kit-API."""

    from omni.kit.viewport.utility import capture_viewport_to_file

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(int(warmup_updates)):
        simulation_app.update()
    last_error = None
    for attempt in range(1, 4):
        if output_path.exists():
            output_path.unlink()
        try:
            capture_helper = capture_viewport_to_file(
                viewport,
                file_path=str(output_path),
                is_hdr=False,
            )
            _wait_for_capture(capture_helper, simulation_app)
            if _wait_for_file(output_path, simulation_app):
                break
            last_error = "Kit-Future abgeschlossen, aber die PNG-Datei fehlt"
        except Exception as exc:
            last_error = str(exc)
        if attempt < 3:
            for _ in range(10):
                simulation_app.update()
    else:
        raise RuntimeError(
            f"PNG-Aufnahme wurde nach drei Versuchen nicht erzeugt: "
            f"{output_path} ({last_error})"
        )
    return {
        "path": str(output_path),
        "file_size_bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }


def capture_presentation_checkpoint(
    simulation_app,
    viewport,
    environments,
    configure_camera,
    output_dir,
    checkpoint,
    views=PRESENTATION_VIEWS,
):
    """Exportiert alle geforderten Ansichten an einem Simulationszeitpunkt."""

    output_dir = Path(output_dir).resolve()
    records = []
    for view_name in views:
        eye, target = camera_pose(view_name, environments)
        configure_camera(view_name)
        output_path = output_dir / render_filename(checkpoint, view_name)
        record = capture_viewport_png(
            simulation_app,
            viewport,
            output_path,
        )
        record.update(
            {
                "checkpoint": str(checkpoint),
                "view": str(view_name),
                "eye_xyz_m": [float(value) for value in eye],
                "target_xyz_m": [float(value) for value in target],
                "resolution": [
                    int(viewport.resolution[0]),
                    int(viewport.resolution[1]),
                ],
            }
        )
        records.append(record)
        print(f"  Rendering gespeichert: {output_path}", flush=True)
    return records


def write_render_manifest(
    path,
    records,
    renderer,
    width,
    height,
    project_version,
):
    """Schreibt einen prüfbaren Nachweis zu allen erzeugten PNG-Dateien."""

    width, height = validate_render_resolution(width, height)
    data = {
        "project_version": str(project_version),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "renderer": str(renderer),
        "resolution": [width, height],
        "image_count": len(records),
        "images": list(records),
        "physics_changed_by_rendering": False,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return data

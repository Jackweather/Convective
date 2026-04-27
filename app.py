from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, send_from_directory


BASE_DIR = Path(os.environ.get("GFS_OUTLOOK_BASE_DIR", "/var/data"))

RUN_DIR_PATTERN = re.compile(r"^gfs_outlook_(\d{8}_\d{2}z)$")
THUNDER_RUN_DIR_PATTERN = re.compile(r"^gfs_thunderstorm_outlook_(\d{8}_\d{2}z)$")

FRAME_PATTERN = re.compile(r"^gfs_outlook_(\d{8}_\d{2}z)_f(\d{3})\.png$")
THUNDER_FRAME_PATTERN = re.compile(r"^gfs_thunderstorm_outlook_(\d{8}_\d{2}z)_f(\d{3})\.png$")


@dataclass(frozen=True)
class FrameInfo:
    file_name: str
    forecast_hour: int
    image_path: str


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    frame_count: int


# ---------------------------
# SCRIPT RUNNERS
# ---------------------------

def run_script(script_path: str, working_dir: str) -> None:
    subprocess.run(
        [sys.executable, script_path],
        cwd=working_dir,
        check=True,
    )


def run_scripts(
    scripts: list[tuple[str, str]],
    retries: int,
    parallel: bool = False,
    max_parallel: int = 3,
) -> None:

    def run_with_retries(script_path: str, working_dir: str) -> None:
        last_error: subprocess.CalledProcessError | None = None

        for _ in range(retries):
            try:
                run_script(script_path, working_dir)
                return
            except subprocess.CalledProcessError as exc:
                last_error = exc

        if last_error:
            raise last_error

    if parallel:
        active_threads: list[threading.Thread] = []

        for script_path, working_dir in scripts:
            thread = threading.Thread(
                target=run_with_retries,
                args=(script_path, working_dir),
                daemon=True,
            )
            thread.start()
            active_threads.append(thread)

            if len(active_threads) >= max_parallel:
                active_threads[0].join()
                active_threads = [
                    t for t in active_threads[1:] if t.is_alive()
                ]

        for thread in active_threads:
            thread.join()
    else:
        for script_path, working_dir in scripts:
            run_with_retries(script_path, working_dir)


# ---------------------------
# DATA COLLECTION FUNCTIONS
# ---------------------------

def collect_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []

    run_dirs = [
        child for child in base_dir.iterdir()
        if RUN_DIR_PATTERN.fullmatch(child.name) and child.is_dir()
    ]

    return sorted(run_dirs, key=lambda p: p.name, reverse=True)


def collect_thunderstorm_run_dirs(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []

    run_dirs = [
        child for child in base_dir.iterdir()
        if THUNDER_RUN_DIR_PATTERN.fullmatch(child.name) and child.is_dir()
    ]

    return sorted(run_dirs, key=lambda p: p.name, reverse=True)


def collect_frames(run_dir: Path) -> list[FrameInfo]:
    png_dir = run_dir / "png"
    if not png_dir.is_dir():
        return []

    frames = []
    for path in png_dir.iterdir():
        match = FRAME_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            frames.append(
                FrameInfo(
                    file_name=path.name,
                    forecast_hour=int(match.group(2)),
                    image_path=f"/images/{run_dir.name}/{path.name}",
                )
            )

    return sorted(frames, key=lambda f: f.forecast_hour)


def collect_thunderstorm_frames(run_dir: Path) -> list[FrameInfo]:
    png_dir = run_dir / "png"
    if not png_dir.is_dir():
        return []

    frames = []
    for path in png_dir.iterdir():
        match = THUNDER_FRAME_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            frames.append(
                FrameInfo(
                    file_name=path.name,
                    forecast_hour=int(match.group(2)),
                    image_path=f"/thunderstorm-images/{run_dir.name}/{path.name}",
                )
            )

    return sorted(frames, key=lambda f: f.forecast_hour)


def collect_runs(base_dir: Path) -> list[RunInfo]:
    return [
        RunInfo(run_name=rd.name, frame_count=len(collect_frames(rd)))
        for rd in collect_run_dirs(base_dir)
    ]


def collect_thunderstorm_runs(base_dir: Path) -> list[RunInfo]:
    return [
        RunInfo(run_name=rd.name, frame_count=len(collect_thunderstorm_frames(rd)))
        for rd in collect_thunderstorm_run_dirs(base_dir)
    ]


def get_latest_run_dir(base_dir: Path) -> Path | None:
    runs = collect_run_dirs(base_dir)
    return runs[0] if runs else None


def build_run_payload(base_dir: Path, run_dir: Path) -> dict:
    frames = collect_frames(run_dir)
    return {
        "run": run_dir.name,
        "baseDir": str(base_dir),
        "frameCount": len(frames),
        "frames": [f.__dict__ for f in frames],
    }


def build_thunderstorm_run_payload(base_dir: Path, run_dir: Path) -> dict:
    frames = collect_thunderstorm_frames(run_dir)
    return {
        "run": run_dir.name,
        "baseDir": str(base_dir),
        "frameCount": len(frames),
        "frames": [f.__dict__ for f in frames],
    }


# ---------------------------
# FLASK APP
# ---------------------------

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["BASE_DIR"] = BASE_DIR

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/run-task1")
    def run_task1():
        scripts = [
            ("/opt/render/project/src/gfs_convective_outlook.py", "/opt/render/project/src"),
            ("/opt/render/project/src/gfs_thunderstorm_outlook.py", "/opt/render/project/src"),
        ]

        threading.Thread(
            target=lambda: run_scripts(scripts, 1, parallel=False),
            daemon=True,
        ).start()

        return "Task started!", 200

    # ---- NORMAL RUNS ----

    @app.get("/api/runs")
    def list_runs():
        return jsonify({
            "baseDir": str(BASE_DIR),
            "runs": [r.__dict__ for r in collect_runs(BASE_DIR)],
        })

    @app.get("/api/runs/latest")
    def latest_run():
        run_dir = get_latest_run_dir(BASE_DIR)
        if not run_dir:
            return jsonify({"run": None}), 404
        return jsonify(build_run_payload(BASE_DIR, run_dir))

    @app.get("/api/runs/<run_name>")
    def run_details(run_name: str):
        run_dir = BASE_DIR / run_name
        if not run_dir.is_dir():
            abort(404)
        return jsonify(build_run_payload(BASE_DIR, run_dir))

    @app.get("/images/<run_name>/<path:file_name>")
    def serve_image(run_name: str, file_name: str):
        return send_from_directory(BASE_DIR / run_name / "png", file_name)

    # ---- THUNDERSTORM RUNS ----

    @app.get("/api/thunderstorm-runs")
    def thunder_runs():
        return jsonify({
            "runs": [r.__dict__ for r in collect_thunderstorm_runs(BASE_DIR)]
        })

    @app.get("/api/thunderstorm-runs/<run_name>")
    def thunder_details(run_name: str):
        run_dir = BASE_DIR / run_name
        if not run_dir.is_dir():
            abort(404)
        return jsonify(build_thunderstorm_run_payload(BASE_DIR, run_dir))

    @app.get("/thunderstorm-images/<run_name>/<path:file_name>")
    def serve_thunder_image(run_name: str, file_name: str):
        return send_from_directory(BASE_DIR / run_name / "png", file_name)

    return app  # 🔥 THIS WAS YOUR BUG




app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

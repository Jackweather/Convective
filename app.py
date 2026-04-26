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
FRAME_PATTERN = re.compile(r"^gfs_outlook_(\d{8}_\d{2}z)_f(\d{3})\.png$")


@dataclass(frozen=True)
class FrameInfo:
    file_name: str
    forecast_hour: int
    image_path: str


@dataclass(frozen=True)
class RunInfo:
    run_name: str
    frame_count: int


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
        if last_error is not None:
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
                active_threads = [item for item in active_threads[1:] if item.is_alive()]

        for thread in active_threads:
            thread.join()
        return

    for script_path, working_dir in scripts:
        run_with_retries(script_path, working_dir)


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
            
        ]
        threading.Thread(
            target=lambda: run_scripts(scripts, 1, parallel=True, max_parallel=3),
            daemon=True,
        ).start()
        return "Task started in background! Check logs folder for output.", 200

    @app.get("/api/runs")
    def list_runs():
        runs = collect_runs(app.config["BASE_DIR"])
        return jsonify(
            {
                "baseDir": str(app.config["BASE_DIR"]),
                "runs": [run.__dict__ for run in runs],
            }
        )

    @app.get("/api/runs/latest")
    def latest_run():
        run_dir = get_latest_run_dir(app.config["BASE_DIR"])
        if run_dir is None:
            return jsonify({"run": None, "frames": [], "baseDir": str(app.config["BASE_DIR"])}), 404
        return jsonify(build_run_payload(app.config["BASE_DIR"], run_dir))

    @app.get("/api/runs/<run_name>")
    def run_details(run_name: str):
        run_dir = app.config["BASE_DIR"] / run_name
        if RUN_DIR_PATTERN.fullmatch(run_name) is None or not run_dir.is_dir():
            abort(404)
        return jsonify(build_run_payload(app.config["BASE_DIR"], run_dir))

    @app.get("/images/<run_name>/<path:file_name>")
    def serve_image(run_name: str, file_name: str):
        run_dir = app.config["BASE_DIR"] / run_name / "png"
        if RUN_DIR_PATTERN.fullmatch(run_name) is None or not run_dir.is_dir():
            abort(404)
        if FRAME_PATTERN.fullmatch(file_name) is None:
            abort(404)
        return send_from_directory(run_dir, file_name)

    return app


def get_latest_run_dir(base_dir: Path) -> Path | None:
    runs = collect_run_dirs(base_dir)
    if not runs:
        return None
    return runs[0]


def collect_run_dirs(base_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if not base_dir.exists():
        return run_dirs

    for child in base_dir.iterdir():
        if RUN_DIR_PATTERN.fullmatch(child.name) is None or not child.is_dir():
            continue
        run_dirs.append(child)

    return sorted(run_dirs, key=lambda path: path.name, reverse=True)


def collect_runs(base_dir: Path) -> list[RunInfo]:
    runs: list[RunInfo] = []
    for run_dir in collect_run_dirs(base_dir):
        runs.append(RunInfo(run_name=run_dir.name, frame_count=len(collect_frames(run_dir))))
    return runs


def build_run_payload(base_dir: Path, run_dir: Path) -> dict[str, object]:
    frames = collect_frames(run_dir)
    return {
        "run": run_dir.name,
        "baseDir": str(base_dir),
        "frameCount": len(frames),
        "frames": [frame.__dict__ for frame in frames],
    }


def collect_frames(run_dir: Path) -> list[FrameInfo]:
    png_dir = run_dir / "png"
    if not png_dir.is_dir():
        return []

    frames: list[FrameInfo] = []
    for path in png_dir.iterdir():
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        forecast_hour = int(match.group(2))
        frames.append(
            FrameInfo(
                file_name=path.name,
                forecast_hour=forecast_hour,
                image_path=f"/images/{run_dir.name}/{path.name}",
            )
        )

    return sorted(frames, key=lambda frame: frame.forecast_hour)


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

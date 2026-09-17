from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import log_lammps_reader
import modal
from loguru import logger
from modal import App, Image

APP_DIR = Path(__file__).resolve().parent if modal.is_local() else Path("/opt/corral-md")
ASSETS = json.loads((APP_DIR / "assets.json").read_text())

lammps_image = (
    Image.debian_slim(python_version=ASSETS["python_version"])
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "liblapack-dev",
        "libfftw3-dev",
        "libopenmpi-dev",
        "openmpi-bin",
        "libssl-dev",
    )
    .pip_install_from_requirements(str(APP_DIR / "requirements.txt"))
    .add_local_file(APP_DIR / "assets.json", "/opt/corral-md/assets.json", copy=True)
    .add_local_file(APP_DIR / "requirements.txt", "/opt/corral-md/requirements.txt", copy=True)
    .run_commands(
        "git init /root/lammps",
        "git -C /root/lammps remote add origin https://github.com/lammps/lammps.git",
        f"git -C /root/lammps fetch --depth 1 origin {ASSETS['lammps_commit']}",
        f"git -C /root/lammps checkout --detach {ASSETS['lammps_commit']}",
        (
            "cmake -S /root/lammps/cmake -B /root/lammps/build "
            "-C /root/lammps/cmake/presets/most.cmake "
            "-C /root/lammps/cmake/presets/nolib.cmake "
            "-DBUILD_MPI=ON -DPKG_MANYBODY=on -DPKG_ATC=yes"
        ),
        "cmake --build /root/lammps/build --parallel 4",
        "cmake --install /root/lammps/build",
        "python -m pip freeze > /opt/corral-md/installed-packages.txt",
    )
)

simagent_name = os.getenv("SIMAGENT_NAME", "")
if simagent_name and not simagent_name.startswith("-"):
    simagent_name = f"-{simagent_name}"

app = App(f"simagent{simagent_name}")
volume_potential = modal.Volume.from_name("potentials", create_if_missing=True)
volume_sim = modal.Volume.from_name("simulations", create_if_missing=True)
volume_struct = modal.Volume.from_name("structures", create_if_missing=True)
volume_models = modal.Volume.from_name("models", create_if_missing=True)
volume_test_files = modal.Volume.from_name("test_files", create_if_missing=True)

CPUS = 2


@app.function(
    image=lammps_image,
    cpu=CPUS,
    timeout=7200,
    memory=5120,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/test_files": volume_test_files,
    },
)
def run_lammps(
    input_file: str,
    log_file: str,
    local_workspace: str | None = None,
    remote_workspace: str | None = None,
) -> None:
    """Run a LAMMPS input from the simulations Volume and persist its outputs."""
    input_path = Path(input_file)
    log_path = input_path.parent / log_file
    original_input: str | None = None
    original_cwd = Path.cwd()

    try:
        volume_sim.reload()
        original_input = input_path.read_text(encoding="utf-8")

        rewritten = original_input
        if local_workspace and remote_workspace:
            rewritten = rewritten.replace(local_workspace, remote_workspace)

        log_command = re.compile(r"^\s*log\s+", re.IGNORECASE)
        sanitized_lines = [
            line
            for line in rewritten.splitlines(keepends=True)
            if line.lstrip().startswith("#") or not log_command.match(line.strip())
        ]
        sanitized_input = "".join(sanitized_lines)
        if sanitized_input != original_input:
            input_path.write_text(sanitized_input, encoding="utf-8")
            volume_sim.commit()

        os.chdir(input_path.parent)
        command = [
            "mpirun",
            "--allow-run-as-root",
            "--bind-to",
            "core",
            "--map-by",
            "core",
            "-np",
            str(CPUS),
            "/root/lammps/build/lmp",
            "-in",
            input_path.name,
            "-log",
            log_file,
        ]
        try:
            subprocess.run(
                command,
                shell=False,
                check=True,
                capture_output=True,
                text=False,
            )
        except subprocess.CalledProcessError:
            if log_path.exists():
                raw_log = log_path.read_bytes()
                log_path.write_text(
                    raw_log.decode("utf-8", errors="ignore"), encoding="utf-8"
                )
            try:
                error_log = log_lammps_reader.log_starts_with(
                    str(log_path), "ERROR"
                )
            except Exception:
                error_log = "Could not read error log"
            raise ValueError(f"LAMMPS simulation failed: {error_log}") from None

        if log_path.exists():
            raw_log = log_path.read_bytes()
            log_path.write_text(
                raw_log.decode("utf-8", errors="ignore"), encoding="utf-8"
            )
        volume_sim.commit()
    except Exception as exc:
        logger.error(f"LAMMPS execution failed: {exc}")
        if log_path.exists():
            raw_log = log_path.read_bytes()
            log_path.write_text(
                raw_log.decode("utf-8", errors="ignore"), encoding="utf-8"
            )
        raise ValueError(str(exc)) from exc
    finally:
        os.chdir(original_cwd)
        if original_input is not None:
            input_path.write_text(original_input, encoding="utf-8")
        volume_sim.commit()

@app.function(
    image=lammps_image,
    gpu="A100",
    cpu=CPUS,
    timeout=7200,
    memory=10240,
    volumes={
        "/potentials": volume_potential.read_only(),
        "/results": volume_sim,
        "/structures": volume_struct.read_only(),
        "/models": volume_models.read_only(),
        "/test_files": volume_test_files,
    },
)
def run_python_gpu(
    script_file: str,
    args: list[str],
    local_workspace: str | None = None,
    remote_workspace: str | None = None,
) -> None:
    """Run a Python script from the simulations Volume with GPU access."""
    script_path = Path(script_file)
    original_script: str | None = None
    original_cwd = Path.cwd()

    try:
        volume_sim.reload()
        original_script = script_path.read_text(encoding="utf-8")

        rewritten = original_script
        if local_workspace and remote_workspace:
            rewritten = rewritten.replace(local_workspace, remote_workspace)
        if rewritten != original_script:
            script_path.write_text(rewritten, encoding="utf-8")
            volume_sim.commit()

        os.chdir(script_path.parent)
        command = [sys.executable, script_path.name, *args]
        result = subprocess.run(
            command, shell=False, check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise ValueError(
                f"GPU script failed (exit {result.returncode}):\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        volume_sim.commit()
    except Exception as exc:
        logger.error(f"GPU script execution failed: {exc}")
        raise ValueError(str(exc)) from exc
    finally:
        os.chdir(original_cwd)
        if original_script is not None:
            script_path.write_text(original_script, encoding="utf-8")
        volume_sim.commit()

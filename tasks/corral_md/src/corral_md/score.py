"""
MD Tutorials Scoring Module

This module provides evaluation and scoring functions for molecular dynamics simulation
tasks and benchmarks. It contains various validation methods to assess the correctness
of simulation results, including numerical comparisons, structural validations, and
analysis of simulation parameters.
"""

# import modal
import hashlib
import json
import pickle
import re
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import modal
import numpy as np
import pandas as pd
from ase import units
from ase.data import atomic_masses, atomic_numbers
from ase.io import read
from loguru import logger
from modal.volume import FileEntryType
from sklearn.metrics import r2_score

PACKAGE_DATA_ROOT = Path(__file__).resolve().parents[2] / "environments"


def _reference_file(value: str | Path) -> Path:
    """Resolve a reference from package data and check it before scoring starts."""
    path = (PACKAGE_DATA_ROOT / value).resolve()
    if path.is_file():
        return path
    mount, _, remote_path = path.as_posix().lstrip("/").partition("/")
    if mount in {"eval_structures", "potentials"}:
        entries = modal.Volume.from_name(mount).listdir(remote_path)
        if any(
            entry.type == FileEntryType.FILE and entry.path.lstrip("/") == remote_path
            for entry in entries
        ):
            return path
    raise FileNotFoundError(f"Reference file not found: {path}")


def check_phonon(target: str, tolerance: float) -> Callable[[Any], float]:
    """
    Create a scoring function that validates the linear regression coefficients
    for the lattice strain vs. optical phonon frequency task.

    The function verifies that:
    1. The submitted JSON file exists and can be parsed.
    2. It contains the keys "isotropic_coefficient" and "uniaxial_coefficient".
    3. Both coefficients match the ground truth values within the relative tolerance.

    Args:
        target (str): Absolute or relative path to the ground truth JSON file.
        tolerance (float): Relative tolerance factor (e.g., 2e-2 for 2%).

    Returns:
        Callable[[Any], float]: A scoring function returning 1.0 if both coefficients pass, else 0.0.
    """

    target = _reference_file(target)

    def score_fn(result: Any) -> float:
        try:
            agent_dict = None

            # 1) Parse the agent's result to extract the dictionary
            if isinstance(result, str):
                s = result.strip()
                # If the string is a file path to a JSON file
                if s.endswith(".json") and Path(s).is_file():
                    try:
                        with open(s) as f:
                            agent_dict = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to read agent JSON file at {s}: {e}")
                        return 0.0
                else:
                    # Try parsing as a raw JSON string
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, dict):
                            # Check if the dict has the keys directly
                            if "isotropic_coefficient" in parsed:
                                agent_dict = parsed
                            else:
                                # Check if it contains a path to the JSON file
                                for v in parsed.values():
                                    if (
                                        isinstance(v, str)
                                        and v.endswith(".json")
                                        and Path(v).is_file()
                                    ):
                                        with open(v) as f:
                                            agent_dict = json.load(f)
                                        break
                    except json.JSONDecodeError:
                        logger.warning(
                            "Result string is neither a valid JSON file path nor a valid JSON string."
                        )
                        return 0.0

            elif isinstance(result, dict):
                # If a dictionary was passed directly
                if "isotropic_coefficient" in result:
                    agent_dict = result
                else:
                    for v in result.values():
                        if (
                            isinstance(v, str)
                            and v.endswith(".json")
                            and Path(v).is_file()
                        ):
                            with open(v) as f:
                                agent_dict = json.load(f)
                            break

            else:
                logger.warning(
                    f"Unrecognized result type in check_phonon: {type(result)}"
                )
                return 0.0

            if not agent_dict or not isinstance(agent_dict, dict):
                logger.warning(
                    "Could not extract a valid dictionary from the agent's submission."
                )
                return 0.0

            # 2) Verify required keys exist in the agent's dictionary
            req_keys = ["isotropic_coefficient", "uniaxial_coefficient"]
            for k in req_keys:
                if k not in agent_dict:
                    logger.warning(f"Missing required key '{k}' in agent's JSON.")
                    return 0.0

            # 3) Load Ground Truth data
            gt_path = Path(target)
            if not gt_path.is_file():
                logger.error(f"Ground truth JSON not found at {gt_path}")
                return 0.0

            try:
                with open(gt_path) as f:
                    gt_dict = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read ground truth JSON: {e}")
                return 0.0

            for k in req_keys:
                if k not in gt_dict:
                    logger.error(f"Ground truth JSON is missing key '{k}'")
                    return 0.0

            # 4) Compare Agent Values vs Ground Truth Values
            for k in req_keys:
                try:
                    agent_val = float(agent_dict[k])
                    gt_val = float(gt_dict[k])
                except (ValueError, TypeError):
                    logger.warning(
                        f"Coefficient for '{k}' could not be converted to float."
                    )
                    return 0.0

                tol = tolerance * abs(gt_val)
                lower_bound = gt_val - tol
                upper_bound = gt_val + tol

                logger.info(
                    f"Checking {k}: Target = {gt_val:.6e}, Agent = {agent_val:.6e}, Allowed Range = [{lower_bound:.6e}, {upper_bound:.6e}]"
                )

                if not (lower_bound <= agent_val <= upper_bound):
                    logger.warning(
                        f"Value for {k} ({agent_val}) is outside the {tolerance * 100}% tolerance window."
                    )
                    return 0.0

            logger.info("Both phonon strain coefficients passed the tolerance check.")
            return 1.0

        except Exception as exc:
            logger.warning(
                f"Unexpected error in check_phonon: {exc}, result was: {result}"
            )
            return 0.0

    return score_fn


def _validate_trajectory(images, requirements, start_step=0):
    """Validate saved MD evidence before using any temperature statistic."""
    stride = requirements["sample_interval"]
    end_step = start_step + requirements["steps"]
    expected_steps = np.arange(start_step, end_step + 1, stride)
    # ASE observers may include the initial frame or start after one interval.
    if len(images) == len(expected_steps) - 1:
        expected_steps = expected_steps[1:]
    if len(images) < 2 or len(images) != len(expected_steps):
        raise ValueError("Trajectory does not cover the required sampling and duration")

    previous = None
    for atoms, step in zip(images, expected_steps, strict=True):
        symbol = requirements["species"]
        if len(atoms) != requirements["atom_count"] or not np.all(
            atoms.numbers == atomic_numbers[symbol]
        ):
            raise ValueError("Unexpected trajectory composition or atom count")
        if atoms.constraints or not np.all(atoms.pbc):
            raise ValueError("Expected unconstrained atoms with periodic boundaries")
        if not np.allclose(
            atoms.cell @ atoms.cell.T,
            np.eye(3) * requirements["cell_length"] ** 2,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("Unexpected simulation cell")
        if not atoms.has("momenta") or not np.allclose(
            atoms.get_masses(), atomic_masses[atomic_numbers[symbol]]
        ):
            raise ValueError("Missing momenta or unexpected atomic masses")
        if atoms.info.get("step") != step or not np.isclose(
            atoms.info.get("time_fs", np.nan),
            step * requirements["timestep_fs"],
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("Missing or incorrect MD step/time metadata")
        if (
            atoms.info.get("model") != requirements["model"]
            or atoms.info.get("thermostat") != "Langevin"
        ):
            raise ValueError("Missing or incorrect model/thermostat metadata")
        if (
            "friction_fs" in requirements
            and atoms.info.get("friction_fs") != requirements["friction_fs"]
        ):
            raise ValueError("Incorrect Langevin friction")
        if atoms.calc is None or atoms.calc.name.lower() not in {
            "mace",
            "macecalculator",
        }:
            raise ValueError("Expected saved MACE calculator results")
        forces = atoms.get_forces()
        if forces.shape != (len(atoms), 3) or not all(
            np.isfinite(value).all()
            for value in (
                atoms.positions,
                atoms.cell,
                atoms.get_momenta(),
                atoms.get_potential_energy(),
                forces,
                atoms.get_temperature(),
            )
        ):
            raise ValueError("Non-finite or malformed trajectory data")
        if previous is not None and (
            np.array_equal(atoms.positions, previous.positions)
            or np.array_equal(atoms.get_momenta(), previous.get_momenta())
        ):
            raise ValueError("Repeated configurations or momenta are not MD dynamics")
        previous = atoms


def _validate_finetuning(result, images, requirements):
    """Check training artifacts and recompute held-out errors, not claimed metrics.

    These are submitted evidence, not independent execution/provenance attestation.
    Never deserialize an agent checkpoint in the grading process.
    """
    submission = json.loads(result) if isinstance(result, str) else result
    checkpoint = Path(submission["checkpoint"])
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError("Missing fine-tuned checkpoint")
    # Modern torch.save checkpoints contain metadata plus tensor storage. Check
    # the container without executing pickle data from a submitted model.
    with ZipFile(checkpoint) as archive:
        entries = {
            entry.filename.partition("/")[2]: entry.file_size
            for entry in archive.infolist()
        }
        if (
            not entries.get("data.pkl")
            or not entries.get("version")
            or not any(
                name.startswith("data/") and size > 0 for name, size in entries.items()
            )
        ):
            raise ValueError("Expected a PyTorch checkpoint with tensor storage")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    checkpoint_hash = digest.hexdigest()
    evaluation = json.loads(Path(submission["evaluation"]).read_text())
    if evaluation["checkpoint_sha256"] != checkpoint_hash or any(
        atoms.info.get("checkpoint_sha256") != checkpoint_hash for atoms in images
    ):
        raise ValueError(
            "Evaluation and trajectory must identify the submitted checkpoint"
        )
    base_hash = evaluation["base_checkpoint_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", base_hash) or base_hash == checkpoint_hash:
        raise ValueError("Checkpoint must differ from the base model")
    losses = np.asarray(evaluation["training_loss"], dtype=float)
    if (
        losses.ndim != 1
        or len(losses) < 2
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError("Missing finite training history")

    dataset = read(submission["dataset"], index=":")
    distances = np.round(np.arange(0.6, 5.01, 0.1), 1)
    if len(dataset) != len(distances):
        raise ValueError("Expected the complete 45-configuration Ag2 dataset")
    energies, forces = [], []
    for atoms, distance in zip(dataset, distances, strict=True):
        if (
            atoms.get_chemical_symbols() != ["Ag", "Ag"]
            or not np.isfinite(atoms.positions).all()
            or not np.isclose(atoms.get_distance(0, 1), distance, rtol=0, atol=1e-6)
        ):
            raise ValueError("Incorrect Ag2 distance grid")
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
    energies, forces = np.asarray(energies), np.asarray(forces)
    if (
        forces.shape != (45, 2, 3)
        or not np.isfinite(energies).all()
        or not np.isfinite(forces).all()
    ):
        raise ValueError("Invalid proxy energy/force labels")
    # Fixed disjoint split; evaluate every fifth distance, in grid order.
    test = np.arange(4, 45, 5)
    train = np.setdiff1d(np.arange(45), test)
    if not np.array_equal(evaluation["train_indices"], train) or not np.array_equal(
        evaluation["test_indices"], test
    ):
        raise ValueError("Incorrect or overlapping training/held-out split")
    for key, reference, limit, divisor in (
        ("energies", energies[test], requirements["energy_rmse_per_atom"], 2),
        ("forces", forces[test], requirements["force_rmse"], 1),
    ):
        prediction = np.asarray(evaluation[key], dtype=float)
        if prediction.shape != reference.shape or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid held-out {key} predictions")
        rmse = np.sqrt(np.mean(((prediction - reference) / divisor) ** 2))
        if not np.isfinite(rmse) or rmse > limit:
            raise ValueError(f"Held-out {key} RMSE exceeds {limit}")


def check_multiple_trajectory_temp(
    target: list[float], tolerance_factor: float, trajectory: dict
) -> Callable[[Any], float]:
    """
    Create a scoring function that validates a dictionary mapping temperatures
    to trajectory files.

    The function verifies that:
    1. The submitted JSON/dict contains keys for all required target temperatures.
    2. Every file path points to an existing .traj file.
    3. Every trajectory satisfies the task's artifact and timing requirements.
    4. The mean temperature and each Cartesian kinetic temperature match the
       target within the specified relative tolerance.

    Args:
        target (List[float]): List of target temperatures (e.g., [300, 400, 500...]).
        tolerance_factor (float): Relative tolerance factor (e.g., 1e-1 for 10%).
        trajectory (dict): Required composition, cell, sampling, and MD metadata.

    Returns:
        Callable[[Any], float]: A scoring function returning 1.0 if all targets pass, else 0.0.
    """

    def score_fn(result: Any) -> float:
        if read is None:
            logger.error("ASE library is not installed. Cannot evaluate .traj files.")
            return 0.0

        try:
            submission_dict = None

            # 1) Parse the agent's result to extract the dictionary
            if isinstance(result, str):
                s = result.strip()
                # If the string is a file path to a JSON file
                if s.endswith(".json") and Path(s).is_file():
                    try:
                        with open(s) as f:
                            submission_dict = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to read JSON file at {s}: {e}")
                        return 0.0
                else:
                    # Try parsing as a raw JSON string
                    try:
                        submission_dict = json.loads(s)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Result string is neither a valid JSON file path nor a valid JSON string."
                        )
                        return 0.0

            elif isinstance(result, dict):
                submission_dict = result
            else:
                logger.warning(
                    f"Unrecognized result type in check_multiple_trajectory_temp: {type(result)}"
                )
                return 0.0

            if not isinstance(submission_dict, dict):
                logger.warning("Parsed result is not a dictionary.")
                return 0.0

            # Normalize keys to strings for robust matching (e.g., handle "300" vs 300)
            normalized_submission = {
                str(k).split(".")[0]: str(v) for k, v in submission_dict.items()
            }

            # 2) Iterate through each required target temperature
            for plateau, t in enumerate(target):
                t_key = str(t).split(".")[0]  # E.g., 300.0 -> '300'

                if t_key not in normalized_submission:
                    logger.warning(
                        f"Missing target temperature {t} K in submitted dictionary."
                    )
                    return 0.0

                traj_path_str = normalized_submission[t_key]
                traj_path = Path(traj_path_str)

                # Verify file exists
                if not traj_path.is_file():
                    logger.warning(f"Trajectory file for {t} K not found: {traj_path}")
                    return 0.0

                # Load trajectory and calculate mean temperature
                try:
                    images = read(traj_path, index=":")
                    start = (
                        plateau
                        * (trajectory["equilibration_steps"] + trajectory["steps"])
                        + trajectory["equilibration_steps"]
                    )
                    _validate_trajectory(images, trajectory, start_step=start)

                    temperatures = [atoms.get_temperature() for atoms in images]
                    mean_temp = float(np.mean(temperatures))
                    # Equipartition requires balanced kinetic energy across axes.
                    component_temperatures = np.mean(
                        [
                            np.mean(
                                atoms.get_momenta() ** 2 / atoms.get_masses()[:, None],
                                axis=0,
                            )
                            / units.kB
                            for atoms in images
                        ],
                        axis=0,
                    )
                    if not np.all(
                        np.abs(component_temperatures - t) <= tolerance_factor * t
                    ):
                        logger.warning(
                            f"Kinetic energy is not equipartitioned at {t} K"
                        )
                        return 0.0
                except Exception as e:
                    logger.warning(f"Failed to read/process trajectory for {t} K: {e}")
                    return 0.0

                # Check if mean temperature falls within the tolerance window
                tol = tolerance_factor * t
                lower_bound = t - tol
                upper_bound = t + tol

                logger.info(
                    f"Target: {t} K | Mean: {mean_temp:.2f} K | Allowed: [{lower_bound:.2f}, {upper_bound:.2f}]"
                )

                if not (lower_bound <= mean_temp <= upper_bound):
                    logger.warning(
                        f"Temperature check failed for {t} K target. Mean temp {mean_temp:.2f} K is out of bounds."
                    )
                    return 0.0

            # If the loop completes successfully, all trajectories passed!
            logger.info("All trajectory temperatures validated successfully.")
            return 1.0

        except Exception as exc:
            logger.warning(
                f"Unexpected error in check_multiple_trajectory_temp: {exc}, result was: {result}"
            )
            return 0.0

    return score_fn


def check_trajectory_temperature(
    target: float,
    window_size: float,
    tolerance: float,
    trajectory: dict,
    finetuning: dict | None = None,
) -> Callable[[Any], float]:
    """
    Create a scoring function that validates a molecular dynamics trajectory file (.traj)
    by calculating the mean temperature of the last N frames.

    Validate task-specific artifacts before checking the final mean temperature.
    Fine-tuning tasks additionally require checkpoint and held-out evaluation artifacts.
    Formula: |mean_temp - target| <= tolerance * abs(target)

    Args:
        target (float): The target temperature in Kelvin (e.g., 300.0).
        window_size (float): The number of frames at the end of the trajectory to average over (e.g., 10.0).
        tolerance (float): Relative tolerance factor (e.g., 1e-1 means +/- 10%).
        trajectory (dict): Required composition, cell, sampling, and MD metadata.
        finetuning (dict | None): Held-out energy and force RMSE limits, when required.

    Returns:
        Callable[[Any], float]: A scoring function that takes a result object
        and returns:
            - 1.0 if the temperature check passes
            - 0.0 otherwise
    """

    def score_fn(result: Any) -> float:
        if read is None:
            logger.error("ASE library is not installed. Cannot evaluate .traj files.")
            return 0.0

        try:
            agent_path_str = None

            # 1) Parse the agent's result to extract the .traj file path
            if isinstance(result, str):
                s = result.strip()
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    # Not JSON: treat the raw string as the file path
                    agent_path_str = s
                else:
                    if isinstance(parsed, dict):
                        agent_path_str = next(
                            (
                                v
                                for v in parsed.values()
                                if isinstance(v, str) and v.endswith(".traj")
                            ),
                            None,
                        )
                        if not agent_path_str and parsed:
                            agent_path_str = str(list(parsed.values())[0])
                    else:
                        agent_path_str = str(parsed)

            elif isinstance(result, dict):
                agent_path_str = next(
                    (
                        v
                        for v in result.values()
                        if isinstance(v, str) and v.endswith(".traj")
                    ),
                    None,
                )
                if not agent_path_str and result:
                    agent_path_str = str(list(result.values())[0])
            else:
                logger.warning(
                    f"Unrecognized result type in check_trajectory_temperature: {type(result)}"
                )
                return 0.0

            if not agent_path_str:
                logger.warning("Could not extract a file path from the result.")
                return 0.0

            # 2) Verify the agent's file exists
            agent_path = Path(agent_path_str)
            if not agent_path.is_file():
                logger.warning(f"Agent's trajectory file not found: {agent_path}")
                return 0.0

            # 3) Load the trajectory using ASE
            try:
                images = read(agent_path, index=":")
                _validate_trajectory(images, trajectory)
                if finetuning is not None:
                    _validate_finetuning(result, images, finetuning)
            except Exception as e:
                logger.warning(f"Failed to read trajectory with ASE: {e}")
                return 0.0

            num_frames = len(images)
            win_size = int(window_size)

            if num_frames == 0:
                logger.warning("Trajectory file is empty.")
                return 0.0

            if win_size < 1 or num_frames < win_size:
                logger.warning(
                    f"Trajectory has {num_frames} frames, but requires a window of {win_size}."
                )
                return 0.0

            # 4) Extract temperatures from the final window
            last_images = images[-win_size:]

            try:
                temperatures = [atoms.get_temperature() for atoms in last_images]
                mean_temp = float(np.mean(temperatures))
            except Exception as e:
                logger.warning(
                    f"Failed to calculate temperature from atoms objects: {e}"
                )
                return 0.0

            # 5) Evaluate against the target and tolerance
            tol = tolerance * abs(target)
            lower_bound = target - tol
            upper_bound = target + tol

            logger.info(
                f"Mean temp (last {win_size} frames): {mean_temp:.2f} K. Allowed range: [{lower_bound:.2f}, {upper_bound:.2f}]"
            )

            if lower_bound <= mean_temp <= upper_bound:
                return 1.0
            else:
                logger.info("Temperature out of bounds.")
                return 0.0

        except Exception as exc:
            logger.warning(
                f"Unexpected error in check_trajectory_temperature: {exc}, result was: {result}"
            )
            return 0.0

    return score_fn


def check_potential_file(target: str):
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior: This is a higher-order function that returns `score_fn`, a callable which:
        - Accepts a single argument `result` (str or None).
        - Logs a warning and returns 0.0 if `result` is None.
        - Otherwise, compares the selected catalog path with the expected path locally.

    Args:
        target (str): The target identifier or path to evaluate results against.

    Returns:
        Callable[[str | None], float]: A function that takes a result string (or None)
        and returns 1.0 for the expected path or 0.0 otherwise.
    """

    target = _reference_file(target)

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_potential_file")
            return 0.0

        try:
            return 1.0 if Path(result).resolve() == Path(target).resolve() else 0.0
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"Could not validate potential path {result!r}: {exc}")
            return 0.0

    return score_fn


def check_log(variable: str | list, target: float, tolerance: float, window: int):
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior:
      - If variable is a string:
          * Extracts that variable from the LAMMPS log file.
          * Computes the average of the last 'window' entries.
          * Compares the average to the target value within tolerance.
      - If variable is a list/tuple [var_to_check, var_must_exist]:
          * Checks var_must_exist exists in the log header.
          * Checks var_to_check numerically as above.
          * If either fails, returns 0.0.
    """
    import json

    import numpy as np

    def read_log_from_text(log_text: str, column: str):
        steps = []
        values = []

        lines = log_text.splitlines()

        header = None
        col_index = None
        step_index = None

        # Allow aliases for certain columns
        column_aliases = {
            "Temp": ["Temp", "Temperature"],
            "Temperature": ["Temp", "Temperature"],
        }

        for raw_line in lines:
            line = raw_line.strip()

            if line.startswith("Step"):
                header = line.split()

                # Resolve column name (handle Temp / Temperature alias)
                possible_names = column_aliases.get(column, [column])

                found_col = None
                for name in possible_names:
                    if name in header:
                        found_col = name
                        break

                if found_col is None:
                    raise ValueError(f"Column '{column}' not found in header: {header}")

                col_index = header.index(found_col)
                step_index = header.index("Step")
                continue

            if header and line:
                tokens = line.split()
                if len(tokens) != len(header):
                    continue
                try:
                    step = int(tokens[step_index])
                    value = float(tokens[col_index])
                except ValueError:
                    continue

                steps.append(step)
                values.append(value)

        return np.array(steps), np.array(values), header

    def header_has_column(header, column: str) -> bool:
        # Handle aliases here too
        column_aliases = {
            "Temp": ["Temp", "Temperature"],
            "Temperature": ["Temp", "Temperature"],
        }
        possible_names = column_aliases.get(column, [column])

        return any(name in header for name in possible_names)

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_log")
            return 0.0
        try:
            data = json.loads(result)
            log_file_path = data["log_file"]

            if "restart_file" in data:
                restart_file_path = Path(data["restart_file"])
                if not restart_file_path.is_file():
                    logger.warning(
                        f"Restart file existence check failed: {restart_file_path}"
                    )
                    return 0.0

            content = Path(log_file_path).read_text(encoding="utf-8")

            # Determine mode
            if isinstance(variable, (list | tuple)):
                var_to_check = variable[0]
                var_must_exist = variable[1]
            else:
                var_to_check = variable
                var_must_exist = None

            steps, values, header = read_log_from_text(content, var_to_check)

            # If second variable must exist, check header
            if var_must_exist is not None and not header_has_column(
                header, var_must_exist
            ):
                logger.warning(
                    f"Required column '{var_must_exist}' not found in log header"
                )
                return 0.0

            if len(values) < window:
                logger.warning(
                    f"Not enough data points ({len(values)}) for the specified window ({window})."
                )
                return 0.0

            recent_values = values[-window:]
            avg_value = np.mean(recent_values)

            tol = tolerance * abs(target)
            if (target - tol) <= avg_value <= (target + tol):
                return 1.0
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Error in check_log: {e}, result was: {result}")
            return 0.0

    return score_fn


def check_msd(target: float):
    """
    Returns a scoring function score_fn(result) -> float in {0.0, 1.0}.

    Behavior:
      - Extracts the MSD value from the result log file.
      - Compares the MSD to the target value within the given tolerance.
    """
    import numpy as np
    from sklearn.metrics import r2_score

    def read_msd_from_text(content: str):
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        if not lines:
            raise ValueError("Empty MSD file")

        def is_float(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        # Detect header: if any token in first line is non-numeric
        first_tokens = lines[0].split()
        has_header = not all(is_float(tok) for tok in first_tokens)

        data_lines = lines[1:] if has_header else lines

        steps = []
        msd = []

        for line in data_lines:
            tokens = line.split()
            if len(tokens) < 2:
                continue
            try:
                step = float(tokens[0])
                val = float(tokens[1])
            except ValueError:
                continue

            steps.append(step)
            msd.append(val)

        if len(msd) == 0:
            raise ValueError("No numeric data found in MSD file")

        return np.array(steps), np.array(msd), has_header

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_msd")
            return 0.0
        try:
            content = Path(result).read_text(encoding="utf-8")
            steps, msd_values, has_header = read_msd_from_text(content)
            # Convert to numpy arrays
            time_ps = np.asarray(steps, dtype=float)
            msd = np.asarray(msd_values, dtype=float)

            # Basic sanity check
            if len(time_ps) < 2:
                return 0.0

            # If you already have a mask logic, keep using it.
            # Otherwise, fit everything:
            mask = np.ones_like(time_ps, dtype=bool)

            time_fit = time_ps[mask]
            msd_fit = msd[mask]

            # Need at least 2 points after masking
            if len(time_fit) < 2:
                return 0.0

            # Linear fit
            slope, intercept = np.polyfit(time_fit, msd_fit, 1)
            msd_fit_line = slope * time_fit + intercept

            # R^2 score
            r2 = r2_score(msd_fit, msd_fit_line)

            logger.info(
                f"MSD fit results: slope={slope}, intercept={intercept}, R^2={r2}"
            )

            # Decision
            if r2 > float(target):
                return 1.0
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error in check_msd: {e}, result was: {result}")
            return 0.0

    return score_fn


def check_numerical(target: float, tolerance: float) -> Callable[[Any], float]:
    """
    Create a scoring function that validates numerical results against a target
    value within a relative tolerance, optionally requiring an associated file
    existence check.

    The returned function evaluates an input `result` and returns a score in
    {0.0, 1.0} according to the following rules:

    - Numeric-only mode:
        * If `result` is a number (int/float), a numeric string, or a JSON
          primitive (number or numeric string), only a numerical tolerance
          check is performed.

    - Dictionary (JSON) mode:
        * If `result` is a dict (or a JSON string that parses to a dict), a
          numerical check is performed AND a file existence check is required.
        * Recognized key pairs:
            - "density"        -> requires "trajectory_file" or "log_file"
            - "BULK ENERGY"    -> requires "path to relaxed structure" or
                                  "Relaxed BULK Structure_path"
            - "SLAB ENERGY"    -> requires "path to relaxed structure" or
                                  "Relaxed BULK Structure_path"

    Numerical validation succeeds if:
        |result - target| <= tolerance * |target|

    Args:
        target (float): The reference numerical value to compare against.
        tolerance (float): Relative tolerance factor applied to `target`.

    Returns:
        Callable[[Any], float]: A scoring function that takes a result object
        (string, number, or dict) and returns:
            - 1.0 if all required numerical (and file, if applicable) checks pass
            - 0.0 otherwise
    """

    def score_fn(result) -> float:
        try:
            answer = None
            file_path = None

            # Helper: numeric-only check
            def numeric_ok(val):
                tol = tolerance * abs(target)
                return (target - tol) <= val <= (target + tol)

            # 1) If input is a string, try JSON first; else try regex numeric
            if isinstance(result, str):
                s = result.strip()
                # Try to parse JSON (could be a dict or a primitive JSON value)
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    # Not JSON: try to extract a number with regex (numeric-only mode)
                    m = re.search(r"[-+]?\d*\.\d+|\d+", s)
                    if not m:
                        return 0.0
                    answer = float(m.group())
                    return 1.0 if numeric_ok(answer) else 0.0
                else:
                    # parsed is the JSON value (dict, number, or string)
                    if isinstance(parsed, dict):
                        parsed_result = parsed
                    else:
                        # primitive JSON value (number or numeric string) -> numeric-only mode
                        try:
                            answer = float(parsed)
                        except (ValueError, TypeError):
                            return 0.0
                        return 1.0 if numeric_ok(answer) else 0.0

            else:
                # result is not a string
                parsed_result = result

            # 2) If we reach here and parsed_result is a dict -> dict-mode
            if isinstance(parsed_result, dict):
                # density -> trajectory_file
                if "density" in parsed_result:
                    try:
                        answer = float(parsed_result["density"])
                    except (ValueError, TypeError):
                        return 0.0
                    file_path = parsed_result.get(
                        "trajectory_file"
                    ) or parsed_result.get("log_file")
                # BULK ENERGY -> path to relaxed structure (or fallback)
                elif "BULK ENERGY" in parsed_result:
                    try:
                        answer = float(parsed_result["BULK ENERGY"])
                    except (ValueError, TypeError):
                        return 0.0
                    file_path = parsed_result.get(
                        "path to relaxed structure"
                    ) or parsed_result.get("Relaxed BULK Structure_path")
                # SLAB ENERGY -> path to relaxed structure (or fallback)
                elif "SLAB ENERGY" in parsed_result:
                    try:
                        answer = float(parsed_result["SLAB ENERGY"])
                    except (ValueError, TypeError):
                        return 0.0
                    file_path = parsed_result.get(
                        "path to relaxed structure"
                    ) or parsed_result.get("Relaxed BULK Structure_path")
                else:
                    # No recognized key present -> fail
                    return 0.0

                # Numeric check
                if not numeric_ok(answer):
                    return 0.0

                # File existence check (required in dict mode)
                if not file_path:
                    logger.warning(
                        "Expected file path key not found in dict submission."
                    )
                    return 0.0

                structure_path = Path(file_path)
                if structure_path.is_file():
                    logger.info(f"Relaxed structure exists: {structure_path}")
                    return 1.0
                logger.warning(f"File existence check failed for {structure_path}")
                return 0.0

            # 3) If parsed_result is a plain numeric (non-string passed in)
            if isinstance(parsed_result, (int | float)):
                answer = float(parsed_result)
                return 1.0 if numeric_ok(answer) else 0.0

            # Fallback: unknown type -> fail
            return 0.0

        except Exception as exc:
            logger.warning(
                f"Unexpected error in check_numerical: {exc}, result was: {result}"
            )
            return 0.0

    return score_fn


def check_structure(target, atom_style):
    """
    Create a scoring function that evaluates a local structure result against a
    reference stored in the read-only `eval_structures` Modal Volume.

    This is a higher-order function that returns `score_fn`, a callable which:
    - Accepts a single argument `result` (str or None).
    - Logs a warning and returns 0.0 if `result` is None.
    - Downloads the small reference structure and compares it to the local result.

    Args:
        target: The target identifier or path used for structure validation.
        atom_style: The LAMMPS atom style used to load both structures.

    Returns:
        Callable[[str | None], float]: A function that takes a result string (or None)
        and returns a floating-point score, or 0.0 if the result is unavailable.
    """

    target = _reference_file(target)

    def score_fn(result: str | None = None) -> float:
        if result is None:
            logger.warning("Received None as result in check_structure")
            return 0.0

        from pymatgen.analysis.structure_matcher import (
            StructureMatcher,
        )
        from pymatgen.io.lammps.data import LammpsData

        result_path = Path(result)
        if not result_path.is_file():
            logger.warning(f"Structure result was not found locally: {result_path}")
            return 0.0

        try:
            with TemporaryDirectory(prefix="corral-md-reference-") as temporary:
                target_path = Path(target)
                if target_path.is_file():
                    local_target = target_path
                elif str(target).startswith("/eval_structures/"):
                    local_target = Path(temporary) / target_path.name
                    volume = modal.Volume.from_name("eval_structures")
                    remote_path = str(target).removeprefix("/eval_structures/")
                    with local_target.open("wb") as stream:
                        for chunk in volume.read_file(remote_path):
                            stream.write(chunk)
                else:
                    logger.warning(f"Structure target was not found: {target}")
                    return 0.0

                expected = LammpsData.from_file(local_target, atom_style=atom_style)
                actual = LammpsData.from_file(result_path, atom_style=atom_style)
                return (
                    1.0
                    if StructureMatcher().fit(expected.structure, actual.structure)
                    else 0.0
                )
        except Exception as exc:
            logger.warning(f"Error in check_structure: {exc}")
            return 0.0

    return score_fn


def _read_spectrum(path):
    data = pd.read_csv(path)
    if list(data.columns) != ["frequency_thz", "vdos"]:
        raise ValueError("Expected exactly frequency_thz and vdos columns")
    values = data.to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            "Spectrum must contain finite, nonnegative frequencies and values"
        )
    frequencies, spectrum = values.T
    if not np.all(np.diff(frequencies) > 0) or not np.any(spectrum > 0):
        raise ValueError("Expected increasing frequencies and a nonzero spectrum")
    return frequencies, spectrum


def check_cosine_similarity(target: str, threshold: float) -> Callable[[Any], float]:
    """
    Create a scoring function that validates a generated spectrum (e.g., VDOS)
    against a set of ground truth spectra using cosine similarity.

    The returned function evaluates an input `result` (which should contain the
    path to a generated CSV file) and returns a score in {0.0, 1.0}.

    Validation requires finite, nonnegative spectra on matching frequency grids,
    and maximum cosine similarity with a ground truth spectrum >= threshold.

    Args:
        target (str): Path to the directory containing ground truth CSV files.
        threshold (float): Minimum cosine similarity score to pass (e.g., 0.90).

    Returns:
        Callable[[Any], float]: A scoring function that takes a result object
        (string, JSON string, or dict) and returns:
            - 1.0 if max cosine similarity >= threshold
            - 0.0 otherwise
    """

    target = (PACKAGE_DATA_ROOT / target).resolve()
    if not any(path.is_file() for path in target.glob("*.csv")):
        raise FileNotFoundError(f"Reference directory must contain CSV files: {target}")

    def score_fn(result: Any) -> float:
        try:
            agent_path_str = None

            # 1) Parse the agent's result to extract the file path
            if isinstance(result, str):
                s = result.strip()
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    # Not JSON: treat the raw string as the file path
                    agent_path_str = s
                else:
                    # Parsed JSON is a dict
                    if isinstance(parsed, dict):
                        # Look for a value ending in .csv
                        agent_path_str = next(
                            (
                                v
                                for v in parsed.values()
                                if isinstance(v, str) and v.endswith(".csv")
                            ),
                            None,
                        )
                        # Fallback: take the first value
                        if not agent_path_str and parsed:
                            agent_path_str = str(list(parsed.values())[0])
                    else:
                        agent_path_str = str(parsed)

            elif isinstance(result, dict):
                # Input is directly a dict
                agent_path_str = next(
                    (
                        v
                        for v in result.values()
                        if isinstance(v, str) and v.endswith(".csv")
                    ),
                    None,
                )
                if not agent_path_str and result:
                    agent_path_str = str(list(result.values())[0])
            else:
                logger.warning(
                    f"Unrecognized result type in check_cosine_similarity: {type(result)}"
                )
                return 0.0

            if not agent_path_str:
                logger.warning("Could not extract a file path from the result.")
                return 0.0

            # 2) Verify the agent's file exists
            agent_path = Path(agent_path_str)
            if not agent_path.is_file():
                logger.warning(f"Agent's CSV file not found: {agent_path}")
                return 0.0

            # 3) Load agent's VDOS data
            try:
                agent_frequencies, agent_vdos = _read_spectrum(agent_path)
            except Exception as e:
                logger.warning(f"Failed to read agent's CSV: {e}")
                return 0.0

            # 4) Find ground truth files
            gt_dir = Path(target)
            gt_files = list(gt_dir.glob("*.csv"))
            if not gt_files:
                logger.error(f"No ground truth CSVs found in target dir: {gt_dir}")
                return 0.0

            # 5) Compute cosine similarity against all ground truths
            max_sim = 0.0
            for gt_file in gt_files:
                try:
                    gt_frequencies, gt_vdos = _read_spectrum(gt_file)
                except Exception as e:
                    logger.warning(f"Failed to read ground truth CSV {gt_file}: {e}")
                    continue

                if agent_frequencies.shape != gt_frequencies.shape or not np.allclose(
                    agent_frequencies, gt_frequencies, rtol=1e-7, atol=1e-8
                ):
                    continue

                # Rescale before normalization to avoid overflow on finite inputs.
                v1 = agent_vdos / np.max(agent_vdos)
                v2 = gt_vdos / np.max(gt_vdos)

                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)

                if norm1 == 0 or norm2 == 0:
                    continue

                sim = np.dot(v1, v2) / (norm1 * norm2)
                if sim > max_sim:
                    max_sim = sim

            logger.info(
                f"Max cosine similarity achieved: {max_sim:.4f} (Threshold: {threshold})"
            )

            # 6) Return final score
            return 1.0 if max_sim >= threshold else 0.0

        except Exception as exc:
            logger.warning(
                f"Unexpected error in check_cosine_similarity: {exc}, result was: {result}"
            )
            return 0.0

    return score_fn


def check_r2(hidden_test_path: str, threshold: float) -> Callable[[Any], float]:
    """
    Create a scoring function that validates a trained regression model (saved as a .pkl)
    by evaluating its R^2 score on a hidden test dataset.

    The returned function extracts the path to a .pkl file from the input `result`,
    loads the model, runs predictions on X_test.npy, and calculates the R^2 score
    against y_test.npy.

    Args:
        hidden_test_path (str): Path to the directory containing 'X_test.npy' and 'y_test.npy'.
        threshold (float): Minimum R^2 score required to pass (e.g., 0.98).

    Returns:
        Callable[[Any], float]: A scoring function that takes a result object
        and returns:
            - 1.0 if R^2 score >= threshold
            - 0.0 otherwise
    """

    hidden_test_path = (PACKAGE_DATA_ROOT / hidden_test_path).resolve()
    for filename in ("X_test.npy", "y_test.npy"):
        _reference_file(hidden_test_path / filename)

    def score_fn(result: Any) -> float:
        try:
            agent_path_str = None

            # 1) Parse the agent's result to extract the .pkl file path
            if isinstance(result, str):
                s = result.strip()
                try:
                    parsed = json.loads(s)
                except json.JSONDecodeError:
                    # Not JSON: treat the raw string as the file path
                    agent_path_str = s
                else:
                    if isinstance(parsed, dict):
                        agent_path_str = next(
                            (
                                v
                                for v in parsed.values()
                                if isinstance(v, str) and v.endswith(".pkl")
                            ),
                            None,
                        )
                        if not agent_path_str and parsed:
                            agent_path_str = str(list(parsed.values())[0])
                    else:
                        agent_path_str = str(parsed)

            elif isinstance(result, dict):
                agent_path_str = next(
                    (
                        v
                        for v in result.values()
                        if isinstance(v, str) and v.endswith(".pkl")
                    ),
                    None,
                )
                if not agent_path_str and result:
                    agent_path_str = str(list(result.values())[0])
            else:
                logger.warning(f"Unrecognized result type in check_r2: {type(result)}")
                return 0.0

            if not agent_path_str:
                logger.warning("Could not extract a file path from the result.")
                return 0.0

            # 2) Verify the agent's model file exists
            agent_path = Path(agent_path_str)
            if not agent_path.is_file():
                logger.warning(f"Agent's model file not found: {agent_path}")
                return 0.0

            # 3) Verify the hidden test data exists
            gt_dir = Path(hidden_test_path)
            x_test_path = gt_dir / "X_test.npy"
            y_test_path = gt_dir / "y_test.npy"

            if not x_test_path.is_file() or not y_test_path.is_file():
                logger.error(
                    f"Hidden test data missing in {gt_dir}. Ensure X_test.npy and y_test.npy exist."
                )
                return 0.0

            # 4) Load hidden test data
            try:
                X_test = np.load(x_test_path)
                y_test = np.load(y_test_path)
            except Exception as e:
                logger.error(f"Failed to load hidden test numpy arrays: {e}")
                return 0.0

            # 5) Load the agent's trained model
            try:
                with open(agent_path, "rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load agent's pickle file: {e}")
                return 0.0

            # 6) Run inference and calculate R^2
            try:
                # Duck-typing check: Ensure the loaded object has a predict method
                if not hasattr(model, "predict"):
                    logger.warning(
                        "Loaded pickle object does not have a 'predict' method."
                    )
                    return 0.0

                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)

            except Exception as e:
                logger.warning(f"Error during model prediction or R^2 calculation: {e}")
                return 0.0

            logger.info(f"R^2 Score achieved: {r2:.4f} (Threshold: {threshold})")

            # 7) Return final score
            return 1.0 if r2 >= threshold else 0.0

        except Exception as exc:
            logger.warning(f"Unexpected error in check_r2: {exc}, result was: {result}")
            return 0.0

    return score_fn

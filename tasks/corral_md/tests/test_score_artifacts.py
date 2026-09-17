"""Artifact contract tests; synthetic fixtures do not attest to a real MD run."""

import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from corral_md import score
from corral_md.submission import resolve_submission


def task_config(level, number):
    path = score.PACKAGE_DATA_ROOT / f"level_{level}/tasks_json/task_{number}.json"
    return json.loads(path.read_text())[0]


def grader(level, number):
    task = task_config(level, number)
    return getattr(score, task["scoring_function"])(**task["scoring_params"])


def frames(requirements, temperature=300, start=0):
    repeats = round((requirements["atom_count"] / 4) ** (1 / 3))
    base = bulk(
        requirements["species"],
        "fcc",
        cubic=True,
        a=requirements["cell_length"] / repeats,
    ).repeat((repeats,) * 3)
    rng = np.random.default_rng(42)
    images = []
    for step in range(
        start, start + requirements["steps"] + 1, requirements["sample_interval"]
    ):
        atoms = base.copy()
        atoms.positions += rng.normal(0, 0.01, atoms.positions.shape)
        momenta = rng.normal(size=atoms.positions.shape)
        momenta *= np.sqrt(
            atoms.get_masses()[:, None]
            * units.kB
            * temperature
            / np.mean(momenta**2, axis=0)
        )
        atoms.set_momenta(momenta)
        atoms.info.update(
            step=step,
            time_fs=step * requirements["timestep_fs"],
            model=requirements["model"],
            thermostat="Langevin",
            friction_fs=0.01,
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=-len(atoms), forces=-atoms.positions * 0.01
        )
        atoms.calc.name = "mace"
        images.append(atoms)
    return images


def save_trajectory(path, images):
    # Rebind synthetic results after mutations to exercise malformed saved data.
    for atoms in images:
        if atoms.calc is not None:
            name, results = atoms.calc.name, atoms.calc.results
            atoms.calc = SinglePointCalculator(atoms, **results)
            atoms.calc.name = name
    write(path, images)
    return str(path)


def test_reported_one_atom_false_positives(tmp_path):
    trajectories = {}
    for temperature in range(300, 901, 100):
        helium = Atoms("He")
        helium.set_momenta(
            [[np.sqrt(3 * units.kB * temperature * helium.get_masses()[0]), 0, 0]]
        )
        trajectories[str(temperature)] = save_trajectory(
            tmp_path / f"{temperature}.traj", [helium]
        )
    assert grader(1, 1)(trajectories["300"]) == 0
    assert grader(2, 3)({"300": trajectories["300"]}) == 0
    assert grader(2, 10)(trajectories) == 0


@pytest.mark.parametrize("include_initial", [True, False])
def test_valid_silver_artifact(tmp_path, include_initial):
    images = frames(task_config(1, 1)["scoring_params"]["trajectory"])
    if not include_initial:
        images = images[1:]
    assert grader(1, 1)(save_trajectory(tmp_path / "silver.traj", images)) == 1


@pytest.mark.parametrize(
    "defect",
    [
        "short",
        "species",
        "atom_count",
        "cell",
        "pbc",
        "masses",
        "momenta",
        "positions_nan",
        "forces_inf",
        "energy_nan",
        "temperature_nan",
        "calculator",
        "missing_forces",
        "step",
        "time",
        "model",
        "thermostat",
        "friction",
        "static",
    ],
)
def test_invalid_silver_artifacts(tmp_path, defect):
    images = frames(task_config(1, 1)["scoring_params"]["trajectory"])
    first = images[0]  # Validate even frames outside the temperature averaging window.
    if defect == "short":
        images = images[:10]
    elif defect == "species":
        first.numbers[0] = 2
    elif defect == "atom_count":
        images[0] = first[:-1]
    elif defect == "cell":
        first.cell *= 2
    elif defect == "pbc":
        first.pbc = False
    elif defect == "masses":
        first.set_masses(np.ones(len(first)))
    elif defect == "momenta":
        del first.arrays["momenta"]
    elif defect == "positions_nan":
        first.positions[0, 0] = np.nan
    elif defect == "forces_inf":
        first.calc.results["forces"][0, 0] = np.inf
    elif defect == "energy_nan":
        first.calc.results["energy"] = np.nan
    elif defect == "temperature_nan":
        first.arrays["momenta"][0, 0] = np.nan
    elif defect == "calculator":
        first.calc.name = "emt"
    elif defect == "missing_forces":
        del first.calc.results["forces"]
    elif defect == "step":
        first.info["step"] = 100
    elif defect == "time":
        first.info["time_fs"] = 1
    elif defect == "model":
        del first.info["model"]
    elif defect == "thermostat":
        first.info["thermostat"] = "NVE"
    elif defect == "friction":
        first.info["friction_fs"] = 1
    elif defect == "static":
        for atoms in images:
            atoms.positions[:] = first.positions
    assert grader(1, 1)(save_trajectory(tmp_path / "invalid.traj", images)) == 0


def test_aluminum_ramp_and_equipartition(tmp_path):
    config = task_config(2, 10)["scoring_params"]
    trajectories = {}
    for plateau, temperature in enumerate(config["target"]):
        images = frames(config["trajectory"], temperature, start=plateau * 5000 + 4000)
        trajectories[str(temperature)] = save_trajectory(
            tmp_path / f"{temperature}.traj", images
        )
    check = grader(2, 10)
    assert check(trajectories) == 1
    # Preserve the total kinetic temperature while putting it all on one axis.
    for atoms in images:
        momenta = atoms.get_momenta()
        momenta[:, 0] *= np.sqrt(3)
        momenta[:, 1:] = 0
        atoms.set_momenta(momenta)
        assert atoms.get_temperature() == pytest.approx(900)
    save_trajectory(tmp_path / "900.traj", images)
    assert check(trajectories) == 0


@pytest.fixture
def finetuning_artifacts(tmp_path):
    # Minimal torch container for structural validation, never deserialized.
    checkpoint = tmp_path / "finetuned.model"
    with ZipFile(checkpoint, "w") as archive:
        for name, value in {
            "data.pkl": b"metadata",
            "version": b"3",
            "data/0": b"tensor",
        }.items():
            archive.writestr(f"model/{name}", value)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    images = frames(task_config(2, 3)["scoring_params"]["trajectory"])
    for atoms in images:
        atoms.info["checkpoint_sha256"] = digest
    dataset = []
    for distance in np.round(np.arange(0.6, 5.01, 0.1), 1):
        atoms = Atoms("Ag2", positions=[[0, 0, 0], [distance, 0, 0]])
        atoms.calc = SinglePointCalculator(
            atoms, energy=distance**2, forces=np.full((2, 3), distance)
        )
        dataset.append(atoms)
    write(tmp_path / "dataset.extxyz", dataset)
    test = np.arange(4, 45, 5)
    evaluation = {
        "checkpoint_sha256": digest,
        "base_checkpoint_sha256": "0" * 64,
        "training_loss": [0.2, 0.1],
        "train_indices": np.setdiff1d(np.arange(45), test).tolist(),
        "test_indices": test.tolist(),
        "energies": [dataset[i].get_potential_energy() for i in test],
        "forces": [dataset[i].get_forces().tolist() for i in test],
    }
    (tmp_path / "evaluation.json").write_text(json.dumps(evaluation))
    return {
        "300": save_trajectory(tmp_path / "300.traj", images),
        "checkpoint": str(checkpoint),
        "dataset": str(tmp_path / "dataset.extxyz"),
        "evaluation": str(tmp_path / "evaluation.json"),
    }


def test_finetuning_artifact_contract(finetuning_artifacts, tmp_path):
    check = grader(2, 3)
    manifest = {
        key: value.rsplit("/", 1)[-1] for key, value in finetuning_artifacts.items()
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert check(resolve_submission("manifest.json", tmp_path)) == 1
    for missing in ("checkpoint", "dataset", "evaluation"):
        assert (
            check(
                {
                    key: value
                    for key, value in finetuning_artifacts.items()
                    if key != missing
                }
            )
            == 0
        )


@pytest.mark.parametrize(
    "defect",
    [
        "checkpoint",
        "hash",
        "base_hash",
        "split",
        "energies",
        "forces",
        "nan",
        "shape",
        "loss",
        "dataset",
    ],
)
def test_invalid_finetuning_evidence(finetuning_artifacts, tmp_path, defect):
    path = tmp_path / "evaluation.json"
    evaluation = json.loads(path.read_text())
    if defect == "checkpoint":
        (tmp_path / "finetuned.model").write_bytes(b"not a checkpoint")
    elif defect == "hash":
        evaluation["checkpoint_sha256"] = "1" * 64
    elif defect == "base_hash":
        evaluation["base_checkpoint_sha256"] = evaluation["checkpoint_sha256"]
    elif defect == "split":
        evaluation["train_indices"].append(evaluation["test_indices"][0])
    elif defect == "energies":
        evaluation["energies"][0] += 100
    elif defect == "forces":
        evaluation["forces"][0][0][0] += 100
    elif defect == "nan":
        evaluation["forces"][0][0][0] = float("nan")
    elif defect == "shape":
        evaluation["energies"].pop()
    elif defect == "loss":
        evaluation["training_loss"] = []
    elif defect == "dataset":
        (tmp_path / "dataset.extxyz").write_text("")
    path.write_text(json.dumps(evaluation))
    assert grader(2, 3)(finetuning_artifacts) == 0


@pytest.mark.parametrize(
    "defect",
    [
        "two_rows",
        "short",
        "long",
        "shifted",
        "reversed",
        "duplicate",
        "negative_frequency",
        "nan_frequency",
        "inf_vdos",
        "negative_vdos",
        "zero",
        "columns",
    ],
)
def test_invalid_vdos_artifacts(tmp_path, defect):
    reference = next(
        (score.PACKAGE_DATA_ROOT / "ground_truth/level_2/task_6").glob("*.csv")
    )
    data = pd.read_csv(reference)
    if defect == "two_rows":
        data = pd.DataFrame({"frequency_thz": ["invalid", "frequency"], "vdos": [0, 1]})
    elif defect == "short":
        data = data.iloc[:2]
    elif defect == "long":
        data.loc[len(data)] = [501, 1]
    elif defect == "shifted":
        data["frequency_thz"] += 0.1
    elif defect == "reversed":
        data = data.iloc[::-1]
    elif defect == "duplicate":
        data.loc[1, "frequency_thz"] = 0
    elif defect == "negative_frequency":
        data.loc[0, "frequency_thz"] = -1
    elif defect == "nan_frequency":
        data.loc[0, "frequency_thz"] = np.nan
    elif defect == "inf_vdos":
        data.loc[0, "vdos"] = np.inf
    elif defect == "negative_vdos":
        data.loc[0, "vdos"] = -1
    elif defect == "zero":
        data["vdos"] = 0
    elif defect == "columns":
        data.columns = ["x", "y"]
    path = tmp_path / "vdos.csv"
    data.to_csv(path, index=False)
    assert grader(2, 6)(str(path)) == 0


def test_vdos_accepts_matching_grid_with_large_finite_scale(tmp_path):
    reference = next(
        (score.PACKAGE_DATA_ROOT / "ground_truth/level_2/task_6").glob("*.csv")
    )
    data = pd.read_csv(reference)
    data["frequency_thz"] += 1e-10
    data["vdos"] *= 1e200
    path = tmp_path / "vdos.csv"
    data.to_csv(path, index=False)
    assert grader(2, 6)(str(path)) == 1

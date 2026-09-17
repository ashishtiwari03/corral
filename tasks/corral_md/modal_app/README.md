# Modal App

The Modal app allows you to run computationally intensive simulation functions on [Modal's](https://modal.com) cloud servers.

## Quick Setup

1. **Create a Modal account** at [modal.com](https://modal.com)
2. **Authenticate**: Run `modal setup` (Modal is already in project dependencies)
3. **Optional**: Set app name via `SIMAGENT_NAME` environment variable

For detailed Modal setup instructions, see the [Modal documentation](https://modal.com/docs/guide).

## MD worker setup and deployment

From `tasks/corral_md`, using the project environment:

```bash
uv run python modal_app/setup_assets.py --upload
uv run modal deploy modal_app/lammps_app.py
```

Setup verifies the bundled ZIP archives and downloads the two exact MACE
checkpoints listed in [assets.json](assets.json), checking SHA-256 before
uploading. It populates the `potentials`, `structures` and `models` Modal
volumes. Repeating setup skips identical remote files and refuses to overwrite
different files. Resolve a reported conflict explicitly before retrying.

To prepare and inspect inputs locally without accessing Modal:

```bash
uv run python modal_app/setup_assets.py --output-dir /tmp/corral-md-assets
```

The GPU worker mounts models read-only at:

- `/models/2023-12-03-mace-128-L1_epoch-199.model`: original medium MACE-MP-0,
  also used to generate proxy labels in the fine-tuning task.
- `/models/mace_agnesi_medium.model`: medium MACE-MP-0b, the fine-tuning starting
  checkpoint. Use the trained checkpoint for that task's subsequent MD.

Use `mace_mp(model=<absolute path>, device="cuda", default_dtype="float64",
dispersion=False)`. Do not rely on the changing default of `mace_mp()`.
The [upstream checkpoint mapping](https://github.com/ACEsuit/mace/blob/v0.3.13/mace/calculators/foundations_models.py)
identifies these two releases. Model binaries are downloaded during setup and
are not committed to this repository.

The worker pins Python 3.12.11, LAMMPS `stable_22Jul2025` by full commit hash,
and all resolved Python dependencies in [requirements.txt](requirements.txt).
To deliberately regenerate the dependency lock, from `modal_app` run:

```bash
uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 --exclude-newer 2025-09-01 -o requirements.txt
```

The manifest and actual installed package versions are retained in the image
at `/opt/corral-md/assets.json` and `/opt/corral-md/installed-packages.txt`.
This records the selected runtime baseline, not the undocumented environment
that produced the existing numerical references. Their targets and tolerances
have not been recalibrated. Record the generation script, random seeds,
manifest and installed-package list when regenerating references. OS packages,
GPU drivers and stochastic trajectories are not made bit-for-bit reproducible
by these pins.

`app.py` is a separate lattice-energy example, not the MD worker; deploy
`lammps_app.py` for this benchmark.

## Using the `@app.function` Decorator in Corral

The `@app.function` decorator defines functions that run on Modal's cloud infrastructure.

### Basic Example

```python
from modal import App, Image

app = App("my-app-name")


@app.function(image=Image.debian_slim().pip_install("numpy"))
def compute_something(data: str) -> float:
    """A function that runs on Modal's servers."""
    import numpy as np

    return np.sum([1, 2, 3, 4, 5])
```

## Usage from Client Code

Call deployed Modal functions from your code:

```python
import modal

# Look up the deployed function
# Note: Both from_name() and lookup() work, this codebase uses from_name()
calculate_lattice_energy = modal.Function.from_name(
    "simagent", "calculate_lattice_energy"
)

# Call the function remotely
energy = calculate_lattice_energy.remote("structure.cif")
```

For more on calling Modal functions, see [Modal's documentation](https://modal.com/docs/guide/call-functions).

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import modal
import pytest
from corral_md.env import create_environments

APP_DIR = Path(__file__).resolve().parents[1] / "modal_app"
spec = importlib.util.spec_from_file_location(
    "md_setup_assets", APP_DIR / "setup_assets.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.fixture
def small_model(monkeypatch):
    payload = b"test checkpoint bytes"
    manifest = {
        **assets.MANIFEST,
        "models": {
            "test.model": {
                "url": "https://example.invalid/test.model",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
    }
    monkeypatch.setattr(assets, "MANIFEST", manifest)
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(payload)
    )
    return payload


def test_prepare_shipped_archives_and_cache_verified_models(
    tmp_path, small_model, monkeypatch
):
    assets.prepare_assets(tmp_path)
    assert (tmp_path / "potentials/SW/Si.sw").is_file()
    assert (tmp_path / "potentials/BKS/pot.mod").is_file()
    assert (tmp_path / "structures/melt/liq4000.dat").is_file()
    assert (tmp_path / "models/test.model").read_bytes() == small_model
    assert not (tmp_path / "__MACOSX").exists()

    def no_download(*_args, **_kwargs):
        pytest.fail("Verified cached models must not be downloaded again")

    monkeypatch.setattr(assets.urllib.request, "urlopen", no_download)
    assets.prepare_assets(tmp_path)


@pytest.mark.usefixtures("small_model")
def test_failed_model_verification_does_not_publish_partial_download(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        assets.urllib.request, "urlopen", lambda *_a, **_kw: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="Checksum mismatch for downloaded model"):
        assets.prepare_assets(tmp_path)
    assert list((tmp_path / "models").iterdir()) == []


def test_archive_verification_rejects_changed_input(tmp_path, monkeypatch):
    (tmp_path / "potentials.zip").write_bytes(b"changed archive")
    monkeypatch.setattr(assets, "TASK_ROOT", tmp_path)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        assets.prepare_assets(tmp_path / "output")


@pytest.mark.parametrize(
    "missing_error", [FileNotFoundError, modal.exception.NotFoundError]
)
@pytest.mark.usefixtures("small_model")
def test_upload_is_repeatable_and_preserves_conflicting_assets(
    tmp_path, monkeypatch, missing_error
):
    assets.prepare_assets(tmp_path)
    files = {}
    uploaded = []

    class Volume:
        def __init__(self, name):
            self.name = name

        def read_file(self, remote):
            if (self.name, remote) not in files:
                raise missing_error(remote)
            yield files[self.name, remote]

        def batch_upload(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def put_file(self, local, remote):
            uploaded.append((self.name, remote))
            files[self.name, remote] = Path(local).read_bytes()

    monkeypatch.setattr(modal.Volume, "from_name", lambda name, **_kw: Volume(name))
    assets.upload_assets(tmp_path)
    assert ("potentials", "/SW/Si.sw") in uploaded
    assert ("models", "/test.model") in uploaded
    uploaded.clear()
    assets.upload_assets(tmp_path)
    assert uploaded == []

    # Even missing files in an earlier volume must not be uploaded if a later
    # volume conflicts with the selected benchmark inputs.
    del files["potentials", "/SW/Si.sw"]
    files["models", "/test.model"] = b"another benchmark's checkpoint"
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        assets.upload_assets(tmp_path)
    assert uploaded == []
    assert files["models", "/test.model"] == b"another benchmark's checkpoint"


def test_prompt_and_task_use_provisioned_checkpoint_paths(tmp_path):
    environments = create_environments(work_dir=str(tmp_path), level=2)
    template = environments["level_2_task_8"]
    bound = template.for_task(task_execution_id="asset-check")
    prompt = bound.initial_event(execution_id="asset-check").task["prompt"]
    manifest = json.loads((APP_DIR / "assets.json").read_text())
    for name in manifest["models"]:
        assert f"/models/{name}" in prompt
    assert "default_dtype='float64'" in prompt
    assert "dispersion=False" in prompt
    assert "./models/" not in bound.current_task.description

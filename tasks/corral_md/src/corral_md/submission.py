"""Resolve MD result files inside the workspace restored for evaluation."""

import json
from pathlib import Path

from corral.workspace import confine_workspace_path


def resolve_submission(submission: str, workspace: str | Path) -> str:
    """Handle numeric answers, result paths, JSON objects, and JSON manifests."""
    root = Path(workspace).resolve()

    def find_file(value: str, base: Path) -> Path:
        supplied = Path(value)
        if ".." in supplied.parts:
            raise ValueError(f"Submitted path cannot traverse its workspace: {value}")
        if not supplied.is_absolute() or supplied.is_relative_to(root):
            for directory in (base, root):
                candidate = confine_workspace_path(root, directory / supplied)
                if candidate.is_file():
                    return candidate

        # Absolute paths may name a former local or Modal workspace. Match the
        # longest trailing path inside the saved workspace, never the host file.
        matches = []
        for candidate in root.rglob("*"):
            if (
                candidate.name != supplied.name
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                continue
            candidate = confine_workspace_path(root, candidate)
            common = 0
            for actual, expected in zip(
                reversed(candidate.relative_to(root).parts),
                reversed(supplied.parts),
                strict=False,
            ):
                if actual != expected:
                    break
                common += 1
            matches.append((common, candidate))
        if not matches:
            raise FileNotFoundError(
                f"Submitted file is missing from the workspace: {value}"
            )
        best = max(length for length, _ in matches)
        paths = [path for length, path in matches if length == best]
        if len(paths) != 1:
            raise ValueError(
                f"Submitted file path is ambiguous in the workspace: {value}"
            )
        return paths[0]

    def resolve_values(value, base: Path):
        if isinstance(value, dict):
            return {key: resolve_values(item, base) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve_values(item, base) for item in value]
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                return str(find_file(value, base))
        return value

    try:
        value = json.loads(submission)
    except json.JSONDecodeError:
        value = submission.strip()
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            pass
        else:
            return submission
        path = find_file(value, root)
        if path.suffix.lower() != ".json":
            return str(path)
        # Paths in a submitted manifest can be relative to its own directory.
        return json.dumps(resolve_values(json.loads(path.read_text()), path.parent))
    return json.dumps(resolve_values(value, root))

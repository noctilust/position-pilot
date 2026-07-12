"""Regression: sdist/wheel packaging excludes caches and Playwright artifacts."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    not (ROOT / "pyproject.toml").exists(),
    reason="repository root not available",
)
def test_sdist_excludes_node_modules_caches_and_playwright_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build an sdist and assert forbidden paths are not packaged."""

    dist = tmp_path / "dist"
    dist.mkdir()
    result = subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(dist)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # Optional fallback when hatch/uv build needs python -m build.
        result = subprocess.run(
            [sys.executable, "-m", "build", "--sdist", "--outdir", str(dist)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr

    archives = list(dist.glob("*.tar.gz"))
    assert archives, "expected an sdist archive"
    archive = archives[0]
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()

    joined = "\n".join(names)
    assert "frontend/node_modules" not in joined
    assert "node_modules/" not in joined
    assert "playwright-report" not in joined
    assert "test-results" not in joined
    assert ".pytest_cache" not in joined
    assert "__pycache__" not in joined
    # Source and docs should still ship.
    assert any("src/position_pilot" in name for name in names)
    assert any(name.endswith("pyproject.toml") for name in names)


def test_wheel_includes_prebuilt_static_when_present(tmp_path: Path) -> None:
    static = ROOT / "src" / "position_pilot" / "web" / "static" / "index.html"
    if not static.exists():
        pytest.skip("static frontend not built in this workspace")

    dist = tmp_path / "dist"
    dist.mkdir()
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(dist.glob("*.whl"))
    assert wheels
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert any("position_pilot/web/static/index.html" in name for name in names)
    assert not any("node_modules" in name for name in names)

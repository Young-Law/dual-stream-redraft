"""Exercise the built distribution outside the checkout, without network access."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_installed_wheel_serves_browser_assets(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    shutil.copy2(root / "pyproject.toml", source)
    shutil.copytree(root / "dualstream", source / "dualstream", ignore=shutil.ignore_patterns("__pycache__"))
    wheels = tmp_path / "wheels"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "--no-index", "-w", str(wheels)],
        cwd=source, check=True, capture_output=True, text=True,
    )
    installed = tmp_path / "installed"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(installed), str(next(wheels.glob("*.whl")))],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    check = '''
from pathlib import Path
from fastapi.testclient import TestClient
import dualstream
from dualstream.api import app
assert Path(dualstream.__file__).resolve().is_relative_to(Path("installed").resolve())
with TestClient(app) as client:
    for path, content in [("/", "DualStream"), ("/static/app.js", "payloadFromForm"),
                          ("/static/api-client.js", "export"), ("/static/styles.css", ".")]:
        response = client.get(path)
        assert response.status_code == 200, (path, response.status_code)
        assert content in response.text, path
'''
    subprocess.run(
        [sys.executable, "-c", check], cwd=tmp_path, check=True,
        env={**os.environ, "PYTHONPATH": str(installed)}, capture_output=True, text=True,
    )

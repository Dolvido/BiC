"""Validate a source checkout and bundled inference demo without training.

Run from the repository root with ``python tools/validate_release.py`` after
installing the project dependencies. Historical experiment suites are deliberately
outside this small, explicitly enumerated release check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = (
    "test_launcher.py",
    "test_offline_home.py",
    "test_home_learning.py",
    "test_definition_tutor_author.py",
    "test_definition_curriculum.py",
    "test_shared_rate_continuation.py",
    "test_shared_rate_run.py",
    "test_public_demo.py",
)
CLI_MODULES = (
    ("brain_in_computer",),
    ("brain_in_computer", "doctor"),
    ("experiments.offline_home",),
    ("experiments.home_learning",),
    ("experiments.shared_rate_run",),
)


def check_syntax() -> int:
    """Compile in memory; avoid writing caches or importing experiments."""
    paths = set(ROOT.glob("*.py"))
    for directory in ("brain_in_computer", "experiments", "tests", "tools", "examples"):
        paths.update((ROOT / directory).rglob("*.py"))
    for path in sorted(paths):
        compile(path.read_bytes(), str(path.relative_to(ROOT)), "exec")
    return len(paths)


def check_import_origins() -> list[str]:
    """Fail if an editable installation supplied code from another checkout."""
    checked = []
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in ("brain_in_computer", "experiments", "examples"):
            continue
        filename = getattr(module, "__file__", None)
        if filename is not None:
            Path(filename).resolve().relative_to(ROOT)
            checked.append(name)
    return sorted(checked)


def check_cli_help() -> list[dict]:
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    records = []
    for arguments in CLI_MODULES:
        command = [sys.executable, "-B", "-m", *arguments, "--help"]
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                text=True, capture_output=True, timeout=60)
        if result.returncode or "usage:" not in result.stdout.lower():
            raise RuntimeError(f"CLI help failed for {arguments}:\n{result.stdout}\n{result.stderr}")
        records.append({"module": arguments[0], "arguments": list(arguments[1:]),
                        "returncode": result.returncode})
    return records


def run_json_tool(script: str, *arguments: str, timeout: int = 30) -> dict:
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    result = subprocess.run([sys.executable, "-B", str(ROOT / script), *arguments],
                            cwd=ROOT, env=environment, text=True, capture_output=True,
                            timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{script} failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional path for a JSON result receipt")
    arguments = parser.parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.dont_write_bytecode = True
    started = time.perf_counter()
    report = {
        "schema": "bic-source-release-validation-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.system(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "Pillow")},
        "test_files": list(TEST_FILES),
        "scope": "Selected source checks, aggregate evidence arithmetic, and bundled CPU inference; no training or live tutor calls.",
    }
    report["syntax_files"] = check_syntax()
    suite = unittest.TestSuite()
    for filename in TEST_FILES:
        selected = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=filename)
        if selected.countTestCases() == 0:
            raise RuntimeError(f"Selected test file has no discoverable tests: {filename}")
        suite.addTests(selected)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report["tests"] = {"run": result.testsRun, "failures": len(result.failures),
                       "errors": len(result.errors), "skipped": len(result.skipped)}
    report["checked_imports"] = check_import_origins()
    report["cli_help"] = check_cli_help()
    evidence = run_json_tool("tools/verify_evidence.py")
    report["evidence"] = {key: evidence[key] for key in (
        "status", "evidence_sha256", "original_report_authenticated",
        "acquisition_mean_effect_pp", "eligible_arms")}
    demo = run_json_tool("examples/english_demo.py", "--json", "--threads", "2")
    report["demo"] = {"device": demo["device"], "weights_sha256": demo["weights_sha256"],
                      "illustrations_not_benchmark": demo["illustrations_not_benchmark"],
                      "scenarios": len(demo["scenarios"]),
                      "turns": sum(len(scenario["turns"]) for scenario in demo["scenarios"])}
    report["passed"] = result.wasSuccessful()
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checked_imports"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

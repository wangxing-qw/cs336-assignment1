"""Local environment checks; webpage declarations are not execution evidence."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

CORE_PACKAGES = ("torch", "numpy", "regex", "einops", "jaxtyping", "psutil", "pytest", "pytest-timeout", "tiktoken", "matplotlib")
COURSE_LOCK_SHA256 = "4f6374b58147216462b76a47eccb58623bdcd9d500e993202493b98dd4d282a0"
EXPECTED = {"python": "3.12.14", "torch": "2.11.0", "numpy": "2.5.3"}
LOCKED_PACKAGES = dict(torch="2.11.0",numpy="2.5.3",regex="2026.9.10",einops="0.8.2",
                      jaxtyping="0.3.11",psutil="7.2.2",pytest="9.1.1",tiktoken="0.14.0",matplotlib="3.11.2")
LOCKED_PACKAGES["pytest-timeout"]="2.4.0"


def collect_environment(lock_path=None):
    import torch
    packages = {}
    for name in CORE_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock = Path(lock_path) if lock_path else Path(__file__).resolve().parents[1]/"uv.lock"
    observed = dict(system=platform.system(), python=platform.python_version(), platform=platform.platform(),
                    machine=platform.machine(), executable=sys.executable, packages=packages,
                    cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None)
    actual = {"python": observed["python"], "torch": (packages["torch"] or "").split("+")[0], "numpy": packages["numpy"]}
    observed["differences"] = {k: {"expected": v, "observed": actual[k]} for k,v in EXPECTED.items() if actual[k] != v}
    observed["teacher_execution_required"] = True
    observed["note"] = "课程要求使用指定Linux及锁定依赖；网页填写属于声明，本地检查不能替代教师评测。"
    return observed


def verify_teacher_runtime(expected_lock_sha256=None, require_cuda=False):
    report = collect_environment()
    errors = list(report["differences"])
    for name,version in LOCKED_PACKAGES.items():
        if (report["packages"].get(name) or "").split("+")[0] != version and name not in errors:
            errors.append(name)
    if expected_lock_sha256 and report["lock_sha256"] != expected_lock_sha256:
        errors.append("uv.lock")
    if require_cuda and (not report["cuda_available"] or report["cuda"] != "12.8"):
        errors.append("CUDA 12.8")
    return dict(passed=not errors, mismatches=errors, environment=report)


def verify_student_runtime(report=None):
    report = collect_environment() if report is None else report
    errors = []
    for name, expected in {"system": "Linux", "python": "3.12.14", "cuda": "12.8", "lock_sha256": COURSE_LOCK_SHA256}.items():
        if report.get(name) != expected:
            errors.append(f"{name}: 要求{expected}，实际{report.get(name)}")
    for name, version in LOCKED_PACKAGES.items():
        actual = report.get("packages", {}).get(name)
        allowed = {version, "2.11.0+cu128"} if name == "torch" else {version}
        if actual not in allowed:
            errors.append(f"{name}: 要求{version}，实际{actual}")
    return dict(passed=not errors, mismatches=errors, environment=report, scope="student_local_only")


def main():
    parser = argparse.ArgumentParser(description="记录环境；--check按课程Linux及锁定版本检查，无需上传报告")
    parser.add_argument("--check", action="store_true", help="发现版本、系统或锁文件不符时以非零状态退出")
    args = parser.parse_args()
    result = verify_student_runtime() if args.check else collect_environment()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.check and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

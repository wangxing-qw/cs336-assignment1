"""Student-side compatibility checks. This is never a teacher acceptance receipt."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from course_tools.contracts import MAX_WEIGHTS_BYTES, sha256, validate_v2_run, validate_weights, write_json,load_weights_file
from course_tools.environment import verify_student_runtime


def preflight(config_path, weights_path, public_manifest, code_dir=None):
    import torch
    environment = verify_student_runtime()
    if not environment["passed"]:
        raise ValueError("本地环境不符合课程要求：" + "; ".join(environment["mismatches"]))
    config = json.loads(Path(config_path).read_text())
    rules = validate_v2_run(config)
    weights = Path(weights_path).resolve()
    if weights.stat().st_size > MAX_WEIGHTS_BYTES:
        raise ValueError("最终权重超过 240 MB")
    state = load_weights_file(weights)
    count = validate_weights(state, rules["model"])
    public = json.loads(Path(public_manifest).read_text())
    report = dict(format_version=2, scope="student_local_only", teacher_accepted=False,
                  passed=True, track=config["track"], parameter_count=count,
                  weights_bytes=weights.stat().st_size, weights_sha256=sha256(weights),
                  config_sha256=sha256(config_path), dataset_id=public["dataset_id"],
                  tokenizer={k:v for k,v in public["files"].items() if k.startswith("tokenizer/")},
                  environment=environment["environment"])
    if code_dir:
        root = Path(code_dir).resolve()
        sys.path[:0] = [str(root), str(root/"src")]
        spec = importlib.util.spec_from_file_location("course_entry", root/"course_entry.py")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        model = entry.load_model(str(weights), config).cpu().float().eval()
        generator = torch.Generator().manual_seed(336)
        ids = torch.randint(0,10000,(2,16), generator=generator)
        changed = ids.clone()
        changed[:,8:] = (changed[:,8:]+7)%10000
        with torch.inference_mode():
            logits = model(ids)
            suffix_changed = model(changed)
        if logits.shape != (2,16,10000) or not torch.isfinite(logits).all():
            raise ValueError("模型前向形状或数值不符合课程接口")
        torch.testing.assert_close(logits[:,:8], suffix_changed[:,:8], atol=1e-5, rtol=1e-4)
        report["student_forward"] = dict(passed=True, logits_shape=list(logits.shape), causality=True)
    return report


def main():
    p = argparse.ArgumentParser()
    for name in ("config", "weights", "public-manifest", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--code-dir")
    a = p.parse_args()
    try:
        result = preflight(a.config, a.weights, a.public_manifest, a.code_dir)
    except Exception as error:
        write_json(a.output, dict(passed=False, teacher_accepted=False, error=str(error)))
        raise SystemExit(str(error))
    write_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

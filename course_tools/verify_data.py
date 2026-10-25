"""Verify the distributed public data; this is not a teacher acceptance receipt."""
import argparse
import json
from pathlib import Path, PurePosixPath

from course_tools.contracts import sha256

PUBLIC_DATA_FILES = (
    "train.txt", "valid.txt", "train.npy", "valid.npy",
    "train.stories.json", "valid.stories.json", "split.json",
    "tokenizer/vocab.json", "tokenizer/merges.json", "tokenizer/tokenizer.json",
)


def safe_path(root, name):
    """Resolve an allowed manifest entry inside the public data directory."""
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or "\\" in name:
        raise ValueError("清单包含非法路径")
    path = Path(root) / parts
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("数据文件不得使用越界路径或符号链接")
    return path


def verify_data(public):
    import numpy as np
    public = Path(public)
    manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest["files"]) != set(PUBLIC_DATA_FILES):
        raise ValueError("公开数据清单的文件集合不符合课程要求")
    for relative, expected in manifest["files"].items():
        if sha256(safe_path(public, relative)) != expected:
            raise ValueError(f"公开数据SHA-256不一致：{relative}")
    if manifest["stories"] != {"train": 100000, "valid": 2000}:
        raise ValueError("公开故事数量不符合课程要求")
    if manifest["tokens"] != {"train": 19683300, "valid": 387711}:
        raise ValueError("公开token数量不符合冻结版本")
    ids = {}
    for split in ("train", "valid"):
        tokens = np.load(public / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        if tokens.dtype != np.uint16 or tokens.ndim != 1 or len(tokens) != manifest["tokens"][split]:
            raise ValueError(f"{split}编码形状、类型或token数量不正确")
        if int(tokens.max()) >= 10000:
            raise ValueError(f"{split}存在超出课程词表的token")
        stories = json.loads((public / f"{split}.stories.json").read_text(encoding="utf-8"))
        ids[split] = {item["sha256"] for item in stories}
        if len(stories) != manifest["stories"][split] or len(ids[split]) != len(stories):
            raise ValueError(f"{split}故事登记数量错误或有完全重复")
    if ids["train"] & ids["valid"]:
        raise ValueError("训练和公开验证之间有完全重复故事")
    return dict(passed=True, teacher_accepted=False, dataset_id=manifest["dataset_id"],
                manifest_sha256=sha256(public / "manifest.json"), stories=manifest["stories"],
                tokens=manifest["tokens"], public_target_tokens=manifest["tokens"]["valid"] - 1,
                checked_files=len(PUBLIC_DATA_FILES), scope="local_public_data_integrity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", default="data/public")
    args = parser.parse_args()
    try:
        print(json.dumps(verify_data(args.public), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()

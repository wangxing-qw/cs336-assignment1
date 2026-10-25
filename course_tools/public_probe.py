"""在只含公开材料的环境运行学生前向；教师另行比较导出模型输出。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    import course_entry
    config = json.loads(Path(a.config).read_text())
    tokens = np.load(a.valid, mmap_mode="r", allow_pickle=False)
    ids = np.array(tokens[:512], dtype=np.int64).reshape(2, 256)
    model = course_entry.load_model(a.checkpoint, config).cpu().float().eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(ids)).detach().cpu().numpy()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"probe.npy", ids, allow_pickle=False)
    np.save(out/"student_logits.npy", logits, allow_pickle=False)


if __name__ == "__main__":
    main()

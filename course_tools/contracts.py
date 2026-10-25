from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

UPSTREAM_COMMIT = "a158843b20107949f1a8d7df1b05cd33b9166712"
DATA_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
EOS = "<|endoftext|>"
MODEL_CONFIG = dict(vocab_size=10000, context_length=256, d_model=512,
                    d_ff=1344, num_layers=4, num_heads=16, rope_theta=10000.0)
PARAMETER_COUNT = 22_696_448
TOKEN_BUDGET = 40_960_000
VALIDATE_EVERY = 4_096_000
MILESTONES = (0, 8_192_000, 24_576_000, 40_960_000)
FORMAT_VERSION = 2
MAX_PARAMETERS = 50_000_000
MAX_WEIGHTS_BYTES = 240_000_000
# The teacher archive retains historical rules; the student bundle includes only
# the rules for this course offering. No model implementation lives here.
TRACKS = json.loads(Path(__file__).with_name("track_rules.json").read_text(encoding="utf-8"))
MODEL_OPTIONS = {"norm": "pre", "position": "rope", "ffn": "swiglu"}


def track_rules(track):
    if track not in TRACKS:
        raise ValueError("不支持的课程档位；允许：" + ", ".join(TRACKS))
    rule = dict(TRACKS[track])
    budget = rule["token_budget"]
    return dict(rule, validate_every=budget//10, milestones=[0, budget//5, budget*3//5, budget],
                ppl_limit=math.exp(rule["nll_limit"]), max_parameters=MAX_PARAMETERS,
                max_weights_bytes=MAX_WEIGHTS_BYTES)


def normalize_model(config=None, *, course=False):
    c = dict(MODEL_CONFIG if config is None else config)
    allowed = set(MODEL_CONFIG) | set(MODEL_OPTIONS)
    if set(c) - allowed or not (set(MODEL_CONFIG) - {"rope_theta"}) <= set(c):
        raise ValueError("模型配置包含未知字段或缺少必要维度")
    c.setdefault("rope_theta", 10000.0)
    for key, value in MODEL_OPTIONS.items():
        c.setdefault(key, value)
    for key in ("vocab_size", "context_length", "d_model", "d_ff", "num_layers", "num_heads"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if any(not isinstance(c[k], str) for k in MODEL_OPTIONS):
        raise ValueError("模型结构选项必须是字符串")
    if c["norm"] not in {"pre", "post", "none"} or c["position"] not in {"rope", "none"} or c["ffn"] not in {"swiglu", "silu"}:
        raise ValueError("不支持的 Transformer 结构选项")
    if c["d_model"] % c["num_heads"]:
        raise ValueError("隐藏维度必须能被注意力头数整除")
    if c["position"] == "rope" and (c["d_model"]//c["num_heads"]) % 2:
        raise ValueError("RoPE 每个头的维度必须为偶数")
    if not isinstance(c["rope_theta"], (int, float)) or not math.isfinite(c["rope_theta"]) or c["rope_theta"] <= 0:
        raise ValueError("非法 RoPE theta")
    if course and (c["vocab_size"] != 10000 or c["context_length"] != 256 or c["rope_theta"] != 10000):
        raise ValueError("课程要求 vocab=10000、context=256、RoPE theta=10000")
    if course and parameter_count(c) > MAX_PARAMETERS:
        raise ValueError("模型总参数超过 50,000,000（包含全部嵌入与输出参数）")
    return c


def parameter_count(config):
    # Arithmetic only: reject oversized configurations before allocating a model.
    c = config
    d, f, layers = c["d_model"], c["d_ff"], c["num_layers"]
    norm = c.get("norm", "pre") != "none"
    projections = 3 if c.get("ffn", "swiglu") == "swiglu" else 2
    return 2*c["vocab_size"]*d + layers*(4*d*d + projections*d*f + (2*d if norm else 0)) + (d if norm else 0)


def validate_v2_run(config):
    if config.get("format_version") != FORMAT_VERSION or config.get("mode") != "ranked":
        raise ValueError("要求 format_version=2、mode=ranked")
    if config.get("ablation", "none") != "none":
        raise ValueError("v2 通过 model.norm/position/ffn 指定结构，不使用旧 ablation 字段")
    model = normalize_model(config["model"], course=True)
    rules = track_rules(config.get("track"))
    for name in ("token_budget", "validate_every", "milestones"):
        if config.get(name) != rules[name]:
            raise ValueError(f"{name} 不符合所选档位")
    quantum, steps = checked_budget(config)
    for name in ("learning_rate","min_learning_rate","eps","weight_decay","grad_clip"):
        value=config.get(name)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError(f"{name}必须是有限非负数")
    if config["learning_rate"]<=0 or config["eps"]<=0 or config["grad_clip"]<=0 or config["min_learning_rate"]>config["learning_rate"]:
        raise ValueError("学习率/epsilon/梯度裁剪参数不合法")
    betas=config.get("betas")
    if not isinstance(betas,(list,tuple)) or len(betas)!=2 or any(type(b) not in (int,float) or not math.isfinite(b) or not 0<=b<1 for b in betas):
        raise ValueError("AdamW betas必须有两个[0,1)参数")
    warmup=config.get("warmup_steps")
    if type(warmup) is not int or not 0<=warmup<steps:
        raise ValueError("warmup_steps须为小于总更新数的非负整数")
    if type(config.get("seed")) is not int or not 0<=config["seed"]<2**32:
        raise ValueError("seed须为32位非负整数")
    if config.get("precision") not in {"fp32","fp16","bf16"}:
        raise ValueError("训练精度须为fp32/fp16/bf16")
    if type(config.get("eval_batch_size")) is not int or config["eval_batch_size"]<=0:
        raise ValueError("eval_batch_size必须为正整数")
    for name in ("train_data","valid_data","data_manifest"):
        if not isinstance(config.get(name),str) or not config[name]:raise ValueError(f"缺少{name}路径")
    return dict(model=model, parameter_count=parameter_count(model), quantum=quantum,
                total_steps=steps, **rules)


def validate_scoring_config(config):
    """Validate a model-only specification, while continuing to read old full configs."""
    if not isinstance(config, dict):
        raise ValueError("模型规格必须为JSON对象")
    if "kind" not in config:
        return validate_v2_run(config)
    if (set(config) != {"format_version", "kind", "track", "model"}
            or config["kind"] != "scoring_model" or config["format_version"] != FORMAT_VERSION):
        raise ValueError("评分模型规格字段或版本不正确")
    model = normalize_model(config["model"], course=True)
    return dict(model=model, parameter_count=parameter_count(model), **track_rules(config["track"]))


def match_training_config(scoring_config, training_config, *, run_id, expected_run_id, selected_step):
    """For later review of the actual experiment config; never invent training values."""
    scoring = validate_scoring_config(scoring_config)
    trained = validate_v2_run(training_config)
    if run_id != expected_run_id:
        raise ValueError("实验目录名与第一阶段登记的最终实验目录不一致")
    if scoring_config["track"] != training_config["track"] or scoring["model"] != trained["model"]:
        raise ValueError("实际训练配置的结构与已验收模型规格不一致")
    if type(selected_step) is not int or not 1 <= selected_step <= trained["total_steps"]:
        raise ValueError("选中模型的训练步数超出该实验的训练预算")
    return trained


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expected_shapes(config=None):
    c = normalize_model(config)
    v, d, f = c["vocab_size"], c["d_model"], c["d_ff"]
    shapes = {"token_embeddings.weight": (v, d), "lm_head.weight": (v, d)}
    if c["norm"] != "none":
        shapes["ln_final.weight"] = (d,)
    for i in range(c["num_layers"]):
        p = f"layers.{i}."
        for name in ("q_proj", "k_proj", "v_proj", "output_proj"):
            shapes[p + f"attn.{name}.weight"] = (d, d)
        if c["norm"] != "none":
            shapes[p + "ln1.weight"] = shapes[p + "ln2.weight"] = (d,)
        for name in (("w1", "w3") if c["ffn"] == "swiglu" else ("w1",)):
            shapes[p + f"ffn.{name}.weight"] = (f, d)
        shapes[p + "ffn.w2.weight"] = (d, f)
    return shapes


def validate_weights(state, config=None):
    import torch
    from collections import OrderedDict
    if type(state) not in (dict, OrderedDict):
        raise ValueError("权重必须是纯 Tensor 字典")
    c = normalize_model(config)
    tensor_count = 2 + (c["norm"] != "none") + c["num_layers"]*(4 + (3 if c["ffn"] == "swiglu" else 2) + (2 if c["norm"] != "none" else 0))
    if len(state) != tensor_count:
        raise ValueError("权重张量数量不匹配")
    shapes = expected_shapes(c)
    if set(state) != set(shapes):
        raise ValueError(f"权重键不匹配: missing={set(shapes)-set(state)}, extra={set(state)-set(shapes)}")
    for key, shape in shapes.items():
        value = state[key]
        if type(value) is not torch.Tensor or tuple(value.shape) != shape or value.layout != torch.strided:
            raise ValueError(f"{key} 的形状应为 {shape}")
        if value.dtype != torch.float32 or not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError(f"{key} 必须是有限、连续的 FP32 权重")
        if value.storage_offset()!=0 or value.untyped_storage().nbytes()!=value.numel()*4:
            raise ValueError(f"{key} 必须独立紧凑存储，不能携带额外storage数据")
    return sum(x.numel() for x in state.values())


def load_weights_file(path):
    """Bound serialized storage before weights-only deserialization; require CPU export."""
    import zipfile
    import torch
    path=Path(path)
    if path.stat().st_size>MAX_WEIGHTS_BYTES:
        raise ValueError("最终权重超过240 MB")
    if not zipfile.is_zipfile(path):
        raise ValueError("请使用默认torch.save生成Tensor字典，不接受旧式或其他格式")
    with zipfile.ZipFile(path) as archive:
        items=archive.infolist()
        if len(items)>200_000 or sum(i.file_size for i in items)>MAX_WEIGHTS_BYTES:
            raise ValueError("权重解封装后超过大小限制")
        if any(i.filename.endswith('/data.pkl') and i.file_size>2_000_000 for i in items):
            raise ValueError("权重元数据异常过大")
    def cpu_only(storage,location):
        if location!='cpu':raise ValueError("评分权重须先导出到CPU")
        return storage
    return torch.load(path,map_location=cpu_only,weights_only=True)


def checked_budget(config):
    for name in ("micro_batch_size", "gradient_accumulation", "token_budget", "validate_every"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} 必须为正整数")
    batch = config["micro_batch_size"] * config["gradient_accumulation"]
    quantum = batch * config["model"]["context_length"]
    budget = config["token_budget"]
    if min(batch, quantum, budget) <= 0 or budget % quantum:
        raise ValueError("训练预算必须是每次优化更新目标 token 数的正整数倍")
    for boundary in [config["validate_every"], *config["milestones"]]:
        if boundary % quantum:
            raise ValueError("验证和过程 checkpoint 边界必须落在优化更新边界")
    return quantum, budget // quantum

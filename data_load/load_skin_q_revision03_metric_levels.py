"""
SKIN-Q 按指标数值分档可视化导出。

对 glossiness / moisture / sebum / elasticity 四个通道分别：
  1) 在当次使用的数据集上统计该指标取值范围；
  2) 划分为 n_bins 个阶段（等宽 uniform 或分位数 quantile）；
  3) 将样本保存到对应指标下、以数值区间命名的子目录（如 1.2345_to_5.6789），
     图下文字仅显示当前分档指标的标签数值。

不修改 load_skin_q.py；复用 revision02 中的绘图与反归一化。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import yaml  # noqa: E402

from load_skin_q import SkinDataLoader  # noqa: E402

from load_skin_q_revision02 import (  # noqa: E402
    CAPTION_FONTSIZE,
    _resolve_raw_item,
    _safe_group_id,
    _safe_stem,
    denormalize_imagenet,
    save_one_sample_figure,
)

# 与 targets 张量下标一致：SkinDataset label CSV 顺序
METRIC_KEYS = ["glossiness", "moisture", "sebum", "elasticity"]


def format_interval_dirname(lo: float, hi: float) -> str:
    """
    分级目录名：直接用区间表示（无 level_xx 前缀）。
    使用 “lo_to_hi” 连接，避免负值或与减号混淆；小数固定 4 位便于阅读与排序。
    """
    return f"{lo:.4f}_to_{hi:.4f}"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_bin_edges(values: np.ndarray, n_bins: int, mode: str) -> np.ndarray:
    """
    返回长度 n_bins+1 的单调递增边界；mode: 'uniform' | 'quantile'。
    全相同或 n_bins<=1 时退化为单区间 [vmin, vmax]。
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n_bins = int(n_bins)
    if len(values) == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    vmin, vmax = float(values.min()), float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return np.array([0.0, 1.0], dtype=np.float64)
    if n_bins <= 1 or abs(vmax - vmin) < 1e-12:
        return np.array([vmin, vmax], dtype=np.float64)

    mode = mode.strip().lower()
    if mode == "uniform":
        return np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)

    if mode == "quantile":
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(values, q).astype(np.float64)
        edges[0] = vmin
        edges[-1] = vmax
        # 分位数重合时微扩，避免空档导致 digitize 异常
        eps = 1e-9 * max(abs(vmax), 1.0)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        return edges

    raise ValueError("bin_mode must be 'uniform' or 'quantile'")


def assign_bin(value: float, edges: np.ndarray) -> int:
    """左闭右开；最后一档右端包含 vmax。"""
    n = len(edges) - 1
    if n <= 0:
        return 0
    v = float(value)
    if v <= edges[0]:
        return 0
    if v >= edges[-1]:
        return n - 1
    b = int(np.searchsorted(edges, v, side="right") - 1)
    return max(0, min(b, n - 1))


def _gather_valid_samples(
    dataset,
    start_index: int,
    end_index: int,
) -> Tuple[List[int], List[np.ndarray]]:
    """返回 (dataset 内下标, 各样本 4 维 targets numpy)。"""
    indices: List[int] = []
    tlist: List[np.ndarray] = []
    for idx in range(start_index, end_index):
        raw = _resolve_raw_item(dataset, idx)
        if _safe_group_id(raw) is None:
            continue
        sample = dataset[idx]
        targets = sample["targets"]
        t = targets.detach().cpu().numpy().astype(np.float64)
        indices.append(idx)
        tlist.append(t)
    return indices, tlist


def _write_metric_summary(
    out_metric_dir: str,
    split_name: str,
    metric_key: str,
    bin_mode: str,
    edges: np.ndarray,
    counts: List[int],
) -> None:
    payload: Dict[str, Any] = {
        "split": split_name,
        "metric": metric_key,
        "bin_mode": bin_mode,
        "n_bins": len(edges) - 1,
        "edges": [float(x) for x in edges],
        "counts_per_bin": counts,
        "total_assigned": int(sum(counts)),
    }
    os.makedirs(out_metric_dir, exist_ok=True)
    with open(os.path.join(out_metric_dir, "bin_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def export_split_by_metric_levels(
    dataset,
    out_root: str,
    split_name: str,
    n_bins: int,
    bin_mode: str,
    use_gmdm: bool,
    use_gwdm: bool,
    max_samples: Optional[int] = None,
    start_index: int = 0,
) -> None:
    n = len(dataset)
    end_cap = n if max_samples is None else min(n, start_index + max_samples)
    valid_indices, targets_list = _gather_valid_samples(dataset, start_index, end_cap)
    if not valid_indices:
        print(f"[{split_name}] no valid samples (group_id); skip stratified export.")
        return

    T = np.stack(targets_list, axis=0)  # [N, 4]

    for mi, metric_key in enumerate(METRIC_KEYS):
        col = T[:, mi]
        edges = compute_bin_edges(col, n_bins=n_bins, mode=bin_mode)
        nb = len(edges) - 1
        counts = [0] * nb

        metric_dir = os.path.join(out_root, split_name, f"by_{metric_key}")
        os.makedirs(metric_dir, exist_ok=True)

        for j, idx in enumerate(valid_indices):
            sample = dataset[idx]
            img = sample["image"]
            targets = sample["targets"]
            filename = sample["filename"]
            if isinstance(filename, (list, tuple)):
                filename = filename[0]

            raw = _resolve_raw_item(dataset, idx)
            group_id = _safe_group_id(raw)
            assert group_id is not None

            val = float(targets[mi].item())
            b = assign_bin(val, edges)
            counts[b] += 1

            lo, hi = float(edges[b]), float(edges[b + 1])
            interval_dir = format_interval_dirname(lo, hi)
            part_dir = os.path.join(metric_dir, interval_dir, f"part_{group_id:02d}")
            stem = _safe_stem(str(filename))
            out_name = f"{idx:05d}_{stem}.png"
            out_path = os.path.join(part_dir, out_name)

            # 图下仅显示当前用做分档的那一维标签数值
            caption = f"{metric_key}: {val:.4f}"

            rgb_np = denormalize_imagenet(img)
            gmdm_hw, gwdm_hw = None, None
            if use_gmdm:
                gm = sample.get("gmdm")
                if gm is not None:
                    gmdm_hw = gm[0].detach().cpu().numpy()
            if use_gwdm:
                gw = sample.get("gwdm")
                if gw is not None:
                    gwdm_hw = gw[0].detach().cpu().numpy()

            save_one_sample_figure(
                rgb_np,
                caption,
                out_path,
                gmdm_hw=gmdm_hw,
                gwdm_hw=gwdm_hw,
                caption_fontsize=CAPTION_FONTSIZE,
            )

        _write_metric_summary(metric_dir, split_name, metric_key, bin_mode, edges, counts)
        print(
            f"[{split_name}][{metric_key}] bins={nb}, mode={bin_mode}, "
            f"counts={counts}, total_saved={sum(counts)}"
        )


def main() -> None:
    default_config = os.path.join(os.path.dirname(_this_dir), "config", "config_skin_Q.yaml")
    config_path = os.environ.get("SKIN_Q_CONFIG", default_config)
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    out_root = os.environ.get("SKIN_Q_REVISION03_OUT")
    if out_root is None:
        cfg = load_config(config_path)
        rp = cfg.get("result_path", "./paper_result/SKIN_Q")
        config_dir = os.path.dirname(os.path.abspath(config_path))
        out_root = os.path.join(
            os.path.abspath(os.path.join(config_dir, rp)),
            "revision03_metric_levels_light02",
        )

    n_bins = int(os.environ.get("SKIN_Q_REVISION03_N_BINS", "4"))
    bin_mode = os.environ.get("SKIN_Q_REVISION03_BIN_MODE", "uniform").strip().lower()

    max_train = os.environ.get("SKIN_Q_REVISION03_MAX_TRAIN")
    max_test = os.environ.get("SKIN_Q_REVISION03_MAX_TEST")
    max_train_i = int(max_train) if max_train else None
    max_test_i = int(max_test) if max_test else None

    print(f"Config: {config_path}")
    print(f"Output: {out_root}")
    print(f"n_bins={n_bins}, bin_mode={bin_mode}")

    loader_setup = SkinDataLoader(config_path, augment=False)

    export_split_by_metric_levels(
        loader_setup.train_dataset,
        out_root,
        "train",
        n_bins=n_bins,
        bin_mode=bin_mode,
        use_gmdm=loader_setup.use_gmdm,
        use_gwdm=loader_setup.use_gwdm,
        max_samples=max_train_i,
    )
    export_split_by_metric_levels(
        loader_setup.test_dataset,
        out_root,
        "test",
        n_bins=n_bins,
        bin_mode=bin_mode,
        use_gmdm=loader_setup.use_gmdm,
        use_gwdm=loader_setup.use_gwdm,
        max_samples=max_test_i,
    )
    print("Done.")


if __name__ == "__main__":
    main()

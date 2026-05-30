"""
SKIN-Q 修订版可视化：逐样本保存图片（非 batch 网格），图片下方标注皮肤指标参数。
加载与预处理逻辑与 load_skin_q.py 一致（含 ImageNet 归一化）；不修改其它模块，仅本脚本。
"""
import os
import sys
import re
import yaml
from typing import Optional

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from load_skin_q import SkinDataLoader  # noqa: E402

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Figure caption (paper-friendly; adjust if needed)
CAPTION_FONTSIZE = 13


def denormalize_imagenet(chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] 反归一化到 [0,1]，返回 HWC float numpy 供 imshow。"""
    x = chw.detach().cpu().float()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = torch.clamp(x, 0.0, 1.0)
    return x.permute(1, 2, 0).numpy()


def _resolve_raw_item(dataset, index: int):
    """从 Dataset / Subset 取原始 json 条目（含 group_id）。"""
    if isinstance(dataset, torch.utils.data.Subset):
        real_idx = int(dataset.indices[index])
        return dataset.dataset.data[real_idx]
    return dataset.data[index]


def _safe_stem(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^\w.\-]+", "_", base)[:120]


def _safe_group_id(raw: dict) -> Optional[int]:
    """JSON 中 group_id 可能缺失或非数字，无法用于分部位目录时返回 None。"""
    gid = raw.get("group_id")
    if gid is None:
        return None
    try:
        return int(gid)
    except (TypeError, ValueError):
        return None


def build_metric_caption(filename: str, description: str, targets: torch.Tensor) -> str:
    # Same order as load_skin_q SkinDataset: label CSV -> [glossiness, moisture, sebum, elasticity]
    g, m, s, el = [float(targets[i].item()) for i in range(4)]
    lines = [
        f"moisture: {m:.2f}    glossiness: {g:.2f}",
        f"sebum: {s:.2f}    elasticity: {el:.2f}",
    ]
    return "\n".join(lines)


def save_one_sample_figure(
    rgb_display: np.ndarray,
    caption: str,
    out_path: str,
    gmdm_hw: Optional[np.ndarray] = None,
    gwdm_hw: Optional[np.ndarray] = None,
    figsize=(5.5, 6.0),
    dpi: int = 150,
    caption_fontsize: int = CAPTION_FONTSIZE,
):
    """
    单张样本：上方为 RGB（可选叠加 GMDM/GWDM），下方为英文指标说明（论文用较大字号）。
    """
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    # 图像贴上方；caption 用 va=top 锚在图下方，减少图与字之间的空白
    ax = fig.add_axes([0.06, 0.34, 0.88, 0.58])
    ax.imshow(rgb_display)
    ax.axis("off")
    if gmdm_hw is not None and gmdm_hw.size > 0:
        ax.imshow(gmdm_hw, cmap="Reds", alpha=0.22)
    if gwdm_hw is not None and gwdm_hw.size > 0:
        ax.imshow(gwdm_hw, cmap="Greens", alpha=0.22)
    # axes 底边约在 y=0.34；caption 顶对齐略低于图底，间距约 0.01–0.02
    fig.text(
        0.5,
        0.325,
        caption,
        ha="center",
        va="top",
        fontsize=caption_fontsize,
        family="sans-serif",
        linespacing=1.2,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def export_split(
    dataset,
    out_root: str,
    split_name: str,
    use_gmdm: bool = False,
    use_gwdm: bool = False,
    max_samples: Optional[int] = None,
    start_index: int = 0,
):
    """
    按样本顺序依次保存；按部位 group_id 分子目录 part_XX。
    group_id 缺失或非法的样本跳过不导出。
    """
    n = len(dataset)
    end = n if max_samples is None else min(n, start_index + max_samples)
    skipped_no_group = 0
    for idx in range(start_index, end):
        sample = dataset[idx]
        img = sample["image"]
        targets = sample["targets"]
        filename = sample["filename"]
        if isinstance(filename, (list, tuple)):
            filename = filename[0]
        desc = sample["description"]
        if isinstance(desc, (list, tuple)):
            desc = desc[0]

        raw = _resolve_raw_item(dataset, idx)
        group_id = _safe_group_id(raw)
        if group_id is None:
            skipped_no_group += 1
            continue

        part_dir = os.path.join(out_root, split_name, f"part_{group_id:02d}")
        stem = _safe_stem(str(filename))
        out_name = f"{idx:05d}_{stem}.png"
        out_path = os.path.join(part_dir, out_name)

        rgb_np = denormalize_imagenet(img)
        caption = build_metric_caption(str(filename), str(desc), targets)

        gmdm_hw, gwdm_hw = None, None
        if use_gmdm:
            gm = sample.get("gmdm")
            if gm is not None:
                gmdm_hw = gm[0].detach().cpu().numpy()
        if use_gwdm:
            gw = sample.get("gwdm")
            if gw is not None:
                gwdm_hw = gw[0].detach().cpu().numpy()

        save_one_sample_figure(rgb_np, caption, out_path, gmdm_hw=gmdm_hw, gwdm_hw=gwdm_hw)
        if (idx - start_index + 1) % 50 == 0:
            print(f"[{split_name}] saved {idx - start_index + 1} / {end - start_index}")

    if skipped_no_group:
        print(
            f"[{split_name}] skipped {skipped_no_group} sample(s) with missing or invalid group_id"
        )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    default_config = os.path.join(os.path.dirname(_this_dir), "config", "config_skin_Q.yaml")
    config_path = os.environ.get("SKIN_Q_CONFIG", default_config)
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    out_root = os.environ.get("SKIN_Q_REVISION02_OUT")
    if out_root is None:
        cfg = load_config(config_path)
        rp = cfg.get("result_path", "./paper_result/SKIN_Q")
        config_dir = os.path.dirname(os.path.abspath(config_path))
        out_root = os.path.join(os.path.abspath(os.path.join(config_dir, rp)), "revision02_vis")

    max_train = os.environ.get("SKIN_Q_REVISION02_MAX_TRAIN")
    max_test = os.environ.get("SKIN_Q_REVISION02_MAX_TEST")
    max_train = int(max_train) if max_train else None
    max_test = int(max_test) if max_test else None

    print(f"Config: {config_path}")
    print(f"Output: {out_root}")

    loader_setup = SkinDataLoader(config_path, augment=False)

    export_split(
        loader_setup.train_dataset,
        out_root,
        "train",
        use_gmdm=loader_setup.use_gmdm,
        use_gwdm=loader_setup.use_gwdm,
        max_samples=max_train,
    )
    export_split(
        loader_setup.test_dataset,
        out_root,
        "test",
        use_gmdm=loader_setup.use_gmdm,
        use_gwdm=loader_setup.use_gwdm,
        max_samples=max_test,
    )

    print("Done.")


if __name__ == "__main__":
    main()

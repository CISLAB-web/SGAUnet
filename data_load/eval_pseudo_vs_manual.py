#!/usr/bin/env python3
"""
Compare gmdm/gwdm pseudo masks to manual *_pores_gt.png / *_wrinkles_gt.png.

- Region: each JSON ROI crop (same box as SkinDataset).
- Macro: arithmetic mean over all evaluated ROIs.
- Pseudo vs full-resolution GT must match RGB H×W (R channel vs binarized GT); else error.
- Binarization: GT L > 127; pseudo uses RGB R channel > 127 (matches mask_transform R slice).

MDM (pores) options (manual marks are often tiny dots):
  --mdm_gt_dilate R   Dilate binary pores GT by R px (disk) before Dice/IoU/P/R vs gmdm.
  --mdm_point_tol R   Per ROI: fraction of manual pore pixels that have gmdm within R px (ellipse).

Example:
  python eval_pseudo_vs_manual.py --config ../config/config_skin_Q.yaml \\
    --gt_light_off ./skin_q_gt_masks_light01 --gt_light_on ./skin_q_gt_masks_light02

  # or pass label/image dirs explicitly (not the YAML path):
  python eval_pseudo_vs_manual.py \\
    --json_folder /path/label --image_folder /path/image \\
    --gt_light_off ./skin_q_gt_masks_light01 --gt_light_on ./skin_q_gt_masks_light02
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml
from PIL import Image


def _crop_box(points):
    left = min(points[0][0], points[1][0])
    top = min(points[0][1], points[1][1])
    right = max(points[0][0], points[1][0])
    bottom = max(points[0][1], points[1][1])
    return left, top, right, bottom


def binarize_u8(arr: np.ndarray) -> np.ndarray:
    return (arr > 127).astype(np.float64)


def binary_metrics_roi(g: np.ndarray, p: np.ndarray) -> dict[str, float]:
    g = g.astype(bool).ravel()
    p = p.astype(bool).ravel()
    inter = np.logical_and(g, p).sum()
    s_g = g.sum()
    s_p = p.sum()
    union = s_g + s_p - inter
    if union == 0:
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    iou = inter / union
    dice = (2.0 * inter / (s_g + s_p)) if (s_g + s_p) > 0 else 0.0
    prec = inter / s_p if s_p > 0 else 0.0
    rec = inter / s_g if s_g > 0 else 0.0
    if s_p == 0 and s_g == 0:
        prec = 1.0
    if s_g == 0 and s_p == 0:
        rec = 1.0
    return {"dice": float(dice), "iou": float(iou), "precision": float(prec), "recall": float(rec)}


def load_gray_bin(path: str) -> np.ndarray:
    return binarize_u8(np.array(Image.open(path).convert("L")))


def load_pseudo_r_bin(path: str) -> np.ndarray:
    rgb = np.array(Image.open(path).convert("RGB"))
    return binarize_u8(rgb[:, :, 0])


def assert_same_hw(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray):
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"Shape mismatch {name_a} {a.shape[:2]} vs {name_b} {b.shape[:2]} "
            "(pseudo and GT must match full image size)"
        )


def _require_label_dir(path: str | None, flag_name: str) -> str:
    if not path:
        return ""
    if os.path.isfile(path) and str(path).lower().endswith((".yaml", ".yml")):
        print(
            f"Error: {flag_name} points to a YAML file ({path!r}).\n"
            "  Use:  --config that_file.yaml   to load paths.json_folder / paths.image_folder\n"
            "  Or:   --json_folder /path/to/label   (the directory of *.json, not config yaml)",
            file=sys.stderr,
        )
        sys.exit(2)
    if not os.path.isdir(path):
        print(f"Error: {flag_name} is not a directory: {path!r}", file=sys.stderr)
        sys.exit(2)
    return path


def build_json_basename_index(json_folder: str) -> dict[str, str]:
    """image basename (e.g. IMG_1.jpg) -> json_path"""
    index = {}
    for fn in sorted(os.listdir(json_folder)):
        if not fn.endswith(".json"):
            continue
        jp = os.path.join(json_folder, fn)
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = os.path.basename(data["imagePath"].replace("\\", "/"))
        if base in index and index[base] != jp:
            raise ValueError(f"Duplicate image basename in JSON: {base}")
        index[base] = jp
    return index


def resolve_image_basename(stem: str, json_index: dict[str, str]) -> str:
    """GT stem (no extension) -> image basename in JSON."""
    for base in json_index:
        if os.path.splitext(base)[0] == stem:
            return base
    raise KeyError(f"No JSON entry whose filename stem matches '{stem}'")


def load_shapes_for_light(json_path: str, light_code: int) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for shape in data.get("shapes", []):
        try:
            parts = shape["description"].split(",")
            if len(parts) != 3:
                continue
            if int(parts[2]) != light_code:
                continue
        except (ValueError, KeyError):
            continue
        out.append(shape)
    return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {k: float("nan") for k in ("dice", "iou", "precision", "recall")}
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def dilate_binary_mask(gp: np.ndarray, radius: int) -> np.ndarray:
    """gp float {0,1}; return float {0,1} after morphological dilation (disk)."""
    if radius <= 0:
        return gp
    u8 = (gp > 0.5).astype(np.uint8) * 255
    k = max(3, 2 * radius + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    d = cv2.dilate(u8, ker)
    return (d > 127).astype(np.float64)


def mdm_point_neighborhood_hit_rate(gp_dots: np.ndarray, pp: np.ndarray, tol_px: int) -> float:
    """
    On undilated manual pore mask (dots): fraction of foreground pixels where pseudo has
    any foreground within Chebyshev/Euclidean neighborhood of radius tol_px.
    Uses dilated pseudo coverage for efficiency.
    """
    g = gp_dots > 0.5
    p = pp > 0.5
    ys, xs = np.where(g)
    if len(ys) == 0:
        return 1.0
    if tol_px <= 0:
        return float(np.sum(p[ys, xs]) / len(ys))
    pu = p.astype(np.uint8) * 255
    k = max(3, 2 * tol_px + 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cover = cv2.dilate(pu, ker) > 127
    hits = np.sum(cover[ys, xs])
    return float(hits / len(ys))


def eval_gt_folder(
    gt_dir: str,
    light_code: int,
    json_index: dict[str, str],
    image_folder: str,
    gmdm_dir: str,
    gwdm_dir: str,
    mdm_gt_dilate: int = 0,
    mdm_point_tol: int = -1,
) -> tuple[list[dict], list[dict], list[str], list[float]]:
    """
    Returns (mdm_per_roi, wdm_per_roi, image_stems, mdm_point_hit_per_roi).
    mdm_point_hit_per_roi is empty unless mdm_point_tol >= 0 (neighborhood hit rate on undilated dots).
    """
    stems = sorted(
        {
            fn.replace("_pores_gt.png", "")
            for fn in os.listdir(gt_dir)
            if fn.endswith("_pores_gt.png")
        }
    )
    mdm_rows: list[dict] = []
    wdm_rows: list[dict] = []
    mdm_point_hits: list[float] = []

    for stem in stems:
        pores_gt_path = os.path.join(gt_dir, f"{stem}_pores_gt.png")
        wr_gt_path = os.path.join(gt_dir, f"{stem}_wrinkles_gt.png")
        if not os.path.isfile(pores_gt_path) or not os.path.isfile(wr_gt_path):
            raise FileNotFoundError(f"Missing pores or wrinkles GT for stem {stem}")

        base = resolve_image_basename(stem, json_index)
        json_path = json_index[base]
        img_path = os.path.join(image_folder, base)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"RGB not found: {img_path}")

        gmdm_path = os.path.join(gmdm_dir, base)
        gwdm_path = os.path.join(gwdm_dir, base)
        if not os.path.isfile(gmdm_path) or not os.path.isfile(gwdm_path):
            raise FileNotFoundError(f"Missing gmdm or gwdm: {base}")

        rgb = np.array(Image.open(img_path).convert("RGB"))
        pores_full = load_gray_bin(pores_gt_path)
        wr_full = load_gray_bin(wr_gt_path)
        gmdm_full = load_pseudo_r_bin(gmdm_path)
        gwdm_full = load_pseudo_r_bin(gwdm_path)

        assert_same_hw("pores_gt", pores_full, "rgb", rgb[:, :, 0])
        assert_same_hw("wrinkles_gt", wr_full, "rgb", rgb[:, :, 0])
        assert_same_hw("gmdm", gmdm_full, "rgb", rgb[:, :, 0])
        assert_same_hw("gwdm", gwdm_full, "rgb", rgb[:, :, 0])

        shapes = load_shapes_for_light(json_path, light_code)
        if not shapes:
            raise RuntimeError(f"No ROI with light_condition={light_code} in {json_path} for {base}")

        for shape in shapes:
            l, t, r, b = _crop_box(shape["points"])
            l, t, r, b = int(l), int(t), int(r), int(b)
            if r <= l or b <= t:
                continue
            gp = pores_full[t:b, l:r]
            pp = gmdm_full[t:b, l:r]
            gw = wr_full[t:b, l:r]
            pw = gwdm_full[t:b, l:r]

            gp_for_mdm = dilate_binary_mask(gp, mdm_gt_dilate)
            mdm_rows.append(binary_metrics_roi(gp_for_mdm, pp))

            if mdm_point_tol >= 0:
                mdm_point_hits.append(mdm_point_neighborhood_hit_rate(gp, pp, mdm_point_tol))

            wdm_rows.append(binary_metrics_roi(gw, pw))

    return mdm_rows, wdm_rows, stems, mdm_point_hits


def main():
    ap = argparse.ArgumentParser(description="Pseudo (gmdm/gwdm) vs manual GT; macro mean over ROIs")
    ap.add_argument(
        "--config",
        default=None,
        help="YAML with paths.json_folder and paths.image_folder (same as training config)",
    )
    ap.add_argument("--json_folder", default=None, help="Label directory containing *.json")
    ap.add_argument("--image_folder", default=None, help="Directory of RGB images")
    ap.add_argument("--gt_light_off", required=True, help="GT folder for light off (e.g. skin_q_gt_masks_light01)")
    ap.add_argument("--gt_light_on", required=True, help="GT folder for light on (e.g. skin_q_gt_masks_light02)")
    ap.add_argument("--light_off_code", type=int, default=1)
    ap.add_argument("--light_on_code", type=int, default=2)
    ap.add_argument("--gmdm_dir", default=None, help="Override gmdm folder (default: sibling of image folder)")
    ap.add_argument("--gwdm_dir", default=None, help="Override gwdm folder")
    ap.add_argument("--out_json", default=None, help="Write metrics JSON here")
    ap.add_argument(
        "--mdm_gt_dilate",
        type=int,
        default=4,
        help="Dilate binary pores GT by this radius (px, ellipse kernel) before MDM Dice/IoU/P/R",
    )
    ap.add_argument(
        "--mdm_point_tol",
        type=int,
        default=5,
        help="If >=0, also report MDM 'point hit rate' per ROI (manual dot has pseudo within R px); macro mean over ROIs",
    )
    args = ap.parse_args()

    json_folder = args.json_folder
    image_folder = args.image_folder
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        json_folder = json_folder or cfg.get("paths", {}).get("json_folder")
        image_folder = image_folder or cfg.get("paths", {}).get("image_folder")
    if not json_folder or not image_folder:
        print(
            "Error: need --config (with paths.json_folder / paths.image_folder) "
            "or both --json_folder and --image_folder.",
            file=sys.stderr,
        )
        sys.exit(2)

    json_folder = _require_label_dir(json_folder, "--json_folder")
    image_folder = _require_label_dir(image_folder, "--image_folder")

    parent = os.path.dirname(os.path.abspath(image_folder))
    gmdm_dir = args.gmdm_dir or os.path.join(parent, "gmdm")
    gwdm_dir = args.gwdm_dir or os.path.join(parent, "gwdm")

    json_index = build_json_basename_index(json_folder)

    mdm_off, wdm_off, stems_off, ph_off = eval_gt_folder(
        args.gt_light_off,
        args.light_off_code,
        json_index,
        image_folder,
        gmdm_dir,
        gwdm_dir,
        mdm_gt_dilate=args.mdm_gt_dilate,
        mdm_point_tol=args.mdm_point_tol,
    )
    mdm_on, wdm_on, stems_on, ph_on = eval_gt_folder(
        args.gt_light_on,
        args.light_on_code,
        json_index,
        image_folder,
        gmdm_dir,
        gwdm_dir,
        mdm_gt_dilate=args.mdm_gt_dilate,
        mdm_point_tol=args.mdm_point_tol,
    )

    mdm_all = mdm_off + mdm_on
    wdm_all = wdm_off + wdm_on
    ph_all = ph_off + ph_on

    agg = {
        "settings": {
            "mdm_gt_dilate_px": args.mdm_gt_dilate,
            "mdm_point_tol_px": args.mdm_point_tol,
        },
        "MDM_Light-off": mean_metrics(mdm_off),
        "MDM_Light-on": mean_metrics(mdm_on),
        "MDM_Overall": mean_metrics(mdm_all),
        "WDM_Light-off": mean_metrics(wdm_off),
        "WDM_Light-on": mean_metrics(wdm_on),
        "WDM_Overall": mean_metrics(wdm_all),
        "counts": {
            "light_off_images": len(stems_off),
            "light_on_images": len(stems_on),
            "light_off_rois": len(mdm_off),
            "light_on_rois": len(mdm_on),
            "overall_images": len(set(stems_off) | set(stems_on)),
            "overall_rois": len(mdm_all),
        },
    }
    if ph_off:
        agg["MDM_point_hit_Light-off"] = float(np.mean(ph_off))
        agg["MDM_point_hit_Light-on"] = float(np.mean(ph_on))
        agg["MDM_point_hit_Overall"] = float(np.mean(ph_all))

    c = agg["counts"]
    print(
        f"=== Macro mean over ROIs (Overall = all ROIs pooled) | "
        f"MDM GT dilate={args.mdm_gt_dilate}px | MDM point-tol={args.mdm_point_tol} ===\n"
    )

    rows_order = [
        ("MDM", "Light-on", agg["MDM_Light-on"], c["light_on_images"], c["light_on_rois"]),
        ("MDM", "Light-off", agg["MDM_Light-off"], c["light_off_images"], c["light_off_rois"]),
        ("MDM", "Overall", agg["MDM_Overall"], c["overall_images"], c["overall_rois"]),
        ("WDM", "Light-on", agg["WDM_Light-on"], c["light_on_images"], c["light_on_rois"]),
        ("WDM", "Light-off", agg["WDM_Light-off"], c["light_off_images"], c["light_off_rois"]),
        ("WDM", "Overall", agg["WDM_Overall"], c["overall_images"], c["overall_rois"]),
    ]
    for mod, light, m, ni, nr in rows_order:
        print(
            f"{mod} & {light:9s} & imgs={ni} ROIs={nr} & "
            f"Dice={m['dice']:.4f} IoU={m['iou']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f}"
        )

    if ph_off:
        print(
            "\n=== MDM point hit rate (each manual pore pixel has pseudo within tol px); macro over ROIs ==="
        )
        print(
            f"MDM_point_hit & Light-on  & imgs={c['light_on_images']} ROIs={c['light_on_rois']} & "
            f"hit={agg['MDM_point_hit_Light-on']:.4f}"
        )
        print(
            f"MDM_point_hit & Light-off & imgs={c['light_off_images']} ROIs={c['light_off_rois']} & "
            f"hit={agg['MDM_point_hit_Light-off']:.4f}"
        )
        print(
            f"MDM_point_hit & Overall   & imgs={c['overall_images']} ROIs={c['overall_rois']} & "
            f"hit={agg['MDM_point_hit_Overall']:.4f}"
        )

    print("\n=== LaTeX (column \\#Images = image count for that row) ===")
    for mod, light, m, ni, nr in rows_order:
        print(
            f"{mod} & {light} & {ni} & {m['dice']:.4f} & {m['iou']:.4f} & "
            f"{m['precision']:.4f} & {m['recall']:.4f} \\\\"
        )

    out_path = args.out_json or os.path.join(
        os.path.abspath(os.path.dirname(args.gt_light_off)), "pseudo_vs_manual_metrics.json"
    )
    out_payload = {**agg, "n_rois_mdm_total": len(mdm_all), "n_rois_wdm_total": len(wdm_all)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SKIN_Q manual labeling: one full-resolution image per file (same layout as gmdm/gwdm on disk).

- Same filters as load_skin_q.SkinDataset (ids, light_conditions); flat items are merged by image_path.
- Brush maps to full image resolution; pores/wrinkles masks are full-size binary PNGs.
- *_gt_meta.json lists all ROI boxes for later crop + Resize IoU.

Tkinter UI: left = overview; right = zoomed ROI panels. Wrinkles use stroke lines.

Examples:
  python annotate_skin_q_roi.py --config ../config/config_skin_Q.yaml --split train --out_dir ./gt_masks
  python annotate_skin_q_roi.py --json_folder /path/label --image_folder /path/image --split full --out_dir ./gt_masks
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import tkinter as tk
import yaml
from PIL import Image, ImageTk

# Same directory as this script so load_skin_q can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_skin_q import SkinDataset  # noqa: E402


def _crop_box(points):
    left = min(points[0][0], points[1][0])
    top = min(points[0][1], points[1][1])
    right = max(points[0][0], points[1][0])
    bottom = max(points[0][1], points[1][1])
    return left, top, right, bottom


def group_flat_items_by_image(ds: SkinDataset):
    """Merge SkinDataset.data rows with the same image_path into one full-image job (rois list matches training crops)."""
    groups: OrderedDict[str, dict] = OrderedDict()
    for item in ds.data:
        key = item["image_path"]
        if key not in groups:
            groups[key] = {
                "image_path": key,
                "json_path": item["json_path"],
                "filename": os.path.basename(key),
                "gmdm_path": item.get("gmdm_path"),
                "gwdm_path": item.get("gwdm_path"),
                "rois": [],
            }
        l, t, r, b = _crop_box(item["points"])
        groups[key]["rois"].append(
            {
                "group_id": item["group_id"],
                "points": item["points"],
                "crop_box": [l, t, r, b],
                "label": item["label"],
                "description": item["description"],
            }
        )
    return list(groups.values())


def load_full_image_job(job: dict):
    """Full RGB + full-size gmdm/gwdm grayscale (same basename as pseudo-label files)."""
    image = Image.open(job["image_path"]).convert("RGB")
    w, h = image.size
    rgb = np.array(image)

    gmdm_g = None
    gwdm_g = None
    gp = job.get("gmdm_path")
    wp = job.get("gwdm_path")
    if gp and os.path.isfile(gp):
        gmdm_g = np.array(Image.open(gp).convert("L"))
        if gmdm_g.shape[0] != h or gmdm_g.shape[1] != w:
            gmdm_g = cv2.resize(gmdm_g, (w, h), interpolation=cv2.INTER_NEAREST)
    if wp and os.path.isfile(wp):
        gwdm_g = np.array(Image.open(wp).convert("L"))
        if gwdm_g.shape[0] != h or gwdm_g.shape[1] != w:
            gwdm_g = cv2.resize(gwdm_g, (w, h), interpolation=cv2.INTER_NEAREST)

    meta = {
        "json_path": job["json_path"],
        "image_path": job["image_path"],
        "filename": job["filename"],
        "image_size_wh": [w, h],
        "rois": job["rois"],
    }
    return rgb, gmdm_g, gwdm_g, meta


def build_dataset(args, cfg: dict | None) -> SkinDataset:
    if cfg is not None:
        jf = cfg["paths"]["json_folder"]
        imf = cfg["paths"]["image_folder"]
    else:
        jf = args.json_folder
        imf = args.image_folder
    # Annotator always resolves gmdm/gwdm paths for overlay (independent of training config use_gmdm)
    use_gmdm = True
    use_gwdm = True

    if args.ids is not None:
        ids = args.ids
    elif cfg is not None and args.split in ("train", "test"):
        ids = cfg["ids"][args.split]
    elif args.split == "full":
        ids = []
    else:
        ids = []

    if args.light_conditions is not None:
        lights = args.light_conditions
    elif cfg is not None and args.split in ("train", "test"):
        lights = cfg["light_conditions"][args.split]
    elif args.split == "full":
        lights = []
    else:
        lights = []

    # Pseudo paths for overlay; GT-only labeling still works if files are missing
    return SkinDataset(
        jf,
        imf,
        ids=ids,
        light_conditions=lights,
        transform=None,
        use_gmdm=use_gmdm,
        use_gwdm=use_gwdm,
    )


def save_paths_full_image(out_dir: str, stem: str) -> tuple[str, str, str]:
    """One file per image (same naming style as gmdm/gwdm; no group_id suffix)."""
    os.makedirs(out_dir, exist_ok=True)
    pores_path = os.path.join(out_dir, f"{stem}_pores_gt.png")
    wr_path = os.path.join(out_dir, f"{stem}_wrinkles_gt.png")
    json_path = os.path.join(out_dir, f"{stem}_gt_meta.json")
    return pores_path, wr_path, json_path


def binarize_mask_u8(m: np.ndarray) -> np.ndarray:
    return (m > 127).astype(np.uint8) * 255


def overlay_visual(rgb: np.ndarray, pores: np.ndarray, wrinkles: np.ndarray, pseudo_gmdm, pseudo_gwdm, show_pseudo: bool):
    """rgb uint8 H×W×3 RGB. GT: magenta=pores, cyan=wrinkles. Pseudo: GMDM bluish, GWDM orange."""
    vis = rgb.astype(np.float32).copy()
    # Optional pseudo masks
    if show_pseudo and pseudo_gmdm is not None:
        g = pseudo_gmdm.astype(np.float32) / 255.0
        tint = np.array([80.0, 120.0, 255.0], dtype=np.float32)  # bluish
        for c in range(3):
            vis[:, :, c] = np.clip(vis[:, :, c] * (1.0 - 0.35 * g) + tint[c] * 0.35 * g, 0, 255)
    if show_pseudo and pseudo_gwdm is not None:
        g = pseudo_gwdm.astype(np.float32) / 255.0
        tint = np.array([255.0, 140.0, 40.0], dtype=np.float32)  # orange
        for c in range(3):
            vis[:, :, c] = np.clip(vis[:, :, c] * (1.0 - 0.35 * g) + tint[c] * 0.35 * g, 0, 255)
    p = pores.astype(np.float32) / 255.0
    w = wrinkles.astype(np.float32) / 255.0
    col_p = np.array([255.0, 0.0, 255.0], dtype=np.float32)  # magenta pores
    col_w = np.array([0.0, 255.0, 255.0], dtype=np.float32)  # cyan wrinkles
    ap, aw = 0.55, 0.55
    for c in range(3):
        vis[:, :, c] = np.clip(
            vis[:, :, c] * (1.0 - ap * p) + col_p[c] * ap * p,
            0,
            255,
        )
    for c in range(3):
        vis[:, :, c] = np.clip(
            vis[:, :, c] * (1.0 - aw * w) + col_w[c] * aw * w,
            0,
            255,
        )
    return vis.astype(np.uint8)


def run_ui(image_jobs: list, start: int, out_dir: str, max_display: int = 960, roi_zoom_max_side: int = 480):
    """Left: overview. Right: zoomed ROIs for drawing. Masks stay full resolution; wrinkles use polylines."""
    idx = start
    n = len(image_jobs)
    if n == 0:
        print("No images to annotate. Check json_folder / ids / light_conditions.")
        return

    mode = "pores"  # 'pores' | 'wrinkles' | 'erase'
    drawing = False
    brush = 8
    show_pseudo = True
    show_rois = True
    stroke_prev = None

    def stem_from_meta(meta):
        return os.path.splitext(meta["filename"])[0]

    def try_load_existing(meta, shape):
        stem = stem_from_meta(meta)
        pp, wp, _ = save_paths_full_image(out_dir, stem)
        pores = np.zeros(shape, dtype=np.uint8)
        wr = np.zeros(shape, dtype=np.uint8)
        if os.path.isfile(pp):
            pores = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
            if pores is None:
                pores = np.zeros(shape, dtype=np.uint8)
            elif pores.shape != shape:
                pores = cv2.resize(pores, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        if os.path.isfile(wp):
            wr = cv2.imread(wp, cv2.IMREAD_GRAYSCALE)
            if wr is None:
                wr = np.zeros(shape, dtype=np.uint8)
            elif wr.shape != shape:
                wr = cv2.resize(wr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return pores, wr

    state = {"rgb": None, "pores": None, "wr": None, "gmdm": None, "gwdm": None, "meta": None, "disp": None, "scale": 1.0}
    photo_ref = []
    roi_panels = []

    def refresh_left_display():
        rgb = state["rgb"]
        if rgb is None:
            return
        h, w = rgb.shape[:2]
        base = rgb.copy()
        if show_rois and state.get("meta"):
            lw = max(1, min(6, max(h, w) // 400))
            for roi in state["meta"].get("rois", []):
                l, t, r, b = [int(x) for x in roi["crop_box"]]
                cv2.rectangle(base, (l, t), (r - 1, b - 1), (0, 255, 0), lw)
        scale = state["scale"]
        state["disp"] = cv2.resize(
            overlay_visual(
                base,
                state["pores"],
                state["wr"],
                state["gmdm"],
                state["gwdm"],
                show_pseudo,
            ),
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    def refresh_roi_panel(panel: dict):
        rgb = state["rgb"]
        if rgb is None:
            return
        l, t, r, b = panel["l"], panel["t"], panel["r"], panel["b"]
        crop_rgb = rgb[t:b, l:r]
        crop_p = state["pores"][t:b, l:r]
        crop_w = state["wr"][t:b, l:r]
        crop_gm = state["gmdm"][t:b, l:r] if state["gmdm"] is not None else None
        crop_gw = state["gwdm"][t:b, l:r] if state["gwdm"] is not None else None
        vis = overlay_visual(crop_rgb, crop_p, crop_w, crop_gm, crop_gw, show_pseudo)
        disp = cv2.resize(vis, (panel["dw"], panel["dh"]), interpolation=cv2.INTER_LINEAR)
        pil_img = Image.fromarray(disp)
        photo = ImageTk.PhotoImage(pil_img)
        photo_ref.append(photo)
        cnv = panel["canvas"]
        cnv.delete("all")
        cnv.config(width=panel["dw"], height=panel["dh"])
        cnv.create_image(0, 0, anchor="nw", image=photo)
        panel["photo"] = photo

    def refresh_all_views():
        refresh_left_display()
        if state["disp"] is None:
            return
        pil_l = Image.fromarray(state["disp"])
        ph_l = ImageTk.PhotoImage(pil_l)
        photo_ref.clear()
        photo_ref.append(ph_l)
        canvas_left.delete("all")
        canvas_left.config(width=state["disp"].shape[1], height=state["disp"].shape[0])
        canvas_left.create_image(0, 0, anchor="nw", image=ph_l)
        for p in roi_panels:
            refresh_roi_panel(p)
        meta = state["meta"]
        if meta:
            nroi = len(meta.get("rois", []))
            status_var.set(
                f"{meta['filename']}  ROIs={nroi}  [{idx + 1}/{n}]  "
                f"mode={mode}  brush={brush}  pseudo={show_pseudo}  boxes={show_rois}  |  draw on right"
            )

    def clamp_global(gx: int, gy: int):
        h, w = state["rgb"].shape[:2]
        return max(0, min(w - 1, gx)), max(0, min(h - 1, gy))

    def paint_global(gx: float, gy: float, prev_xy, is_new_stroke: bool):
        nonlocal mode, brush
        if state["rgb"] is None:
            return
        gx, gy = clamp_global(int(round(gx)), int(round(gy)))
        pores = state["pores"]
        wr = state["wr"]
        r = max(1, int(round(brush)))
        th = max(1, min(64, r))
        # Pores: small dots ~ brush/4 at full resolution
        r_pore = max(1, min(10, int(round(brush / 4.0))))
        if mode == "pores":
            cv2.circle(pores, (gx, gy), r_pore, 255, -1)
        elif mode == "wrinkles":
            if prev_xy is not None and not is_new_stroke:
                x0, y0 = int(prev_xy[0]), int(prev_xy[1])
                x0, y0 = clamp_global(x0, y0)
                cv2.line(wr, (x0, y0), (gx, gy), 255, thickness=th, lineType=cv2.LINE_AA)
            else:
                cv2.circle(wr, (gx, gy), max(1, th // 2), 255, -1)
        else:
            if prev_xy is not None and not is_new_stroke:
                x0, y0 = int(prev_xy[0]), int(prev_xy[1])
                x0, y0 = clamp_global(x0, y0)
                cv2.line(pores, (x0, y0), (gx, gy), 0, thickness=th, lineType=cv2.LINE_AA)
                cv2.line(wr, (x0, y0), (gx, gy), 0, thickness=th, lineType=cv2.LINE_AA)
            else:
                cv2.circle(pores, (gx, gy), r, 0, -1)
                cv2.circle(wr, (gx, gy), r, 0, -1)
        refresh_all_views()

    def canvas_local_to_global(panel: dict, ex: float, ey: float):
        l, t, r, b = panel["l"], panel["t"], panel["r"], panel["b"]
        sc = panel["scale"]
        gx = l + int(round(ex / sc))
        gy = t + int(round(ey / sc))
        gx = max(l, min(r - 1, gx))
        gy = max(t, min(b - 1, gy))
        return float(gx), float(gy)

    def make_roi_mouse_handlers(panel: dict):
        def on_press(e):
            nonlocal drawing, stroke_prev
            drawing = True
            stroke_prev = None
            gx, gy = canvas_local_to_global(panel, e.x, e.y)
            paint_global(gx, gy, None, is_new_stroke=True)
            stroke_prev = (gx, gy)

        def on_motion(e):
            nonlocal stroke_prev
            if not drawing:
                return
            gx, gy = canvas_local_to_global(panel, e.x, e.y)
            paint_global(gx, gy, stroke_prev, is_new_stroke=False)
            stroke_prev = (gx, gy)

        def on_release(_e):
            nonlocal drawing, stroke_prev
            drawing = False
            stroke_prev = None

        return on_press, on_motion, on_release

    root = tk.Tk()
    root.title("SKIN_Q annotate — overview | zoomed ROIs")
    status_var = tk.StringVar(value="")

    help_txt = (
        "Draw on right | [p]pores (dots ~brush/4) [l]wrinkles [e]erase [o]pseudo [v]ROI boxes [+/-]brush "
        "[n]next [b]prev [s]save [c]clear [q]quit"
    )
    print(help_txt)

    top = tk.Label(root, textvariable=status_var, justify="left", font=("TkFixedFont", 10))
    top.pack(fill="x", padx=4, pady=2)

    pw = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
    pw.pack(fill="both", expand=True, padx=4, pady=2)

    left_col = tk.Frame(pw, bg="#222222")
    right_col = tk.Frame(pw, bg="#1a1a1a")
    pw.add(left_col, minsize=240)
    pw.add(right_col, minsize=320)

    tk.Label(left_col, text="Full image (preview only)", fg="#aaa", bg="#222222").pack(anchor="w")
    canvas_left = tk.Canvas(left_col, bg="#222222", highlightthickness=0)
    canvas_left.pack(fill="both", expand=True)

    tk.Label(right_col, text="Zoomed ROIs (draw here)", fg="#aaa", bg="#1a1a1a").pack(anchor="w")
    scroll_wrap = tk.Frame(right_col, bg="#1a1a1a")
    scroll_wrap.pack(fill="both", expand=True)
    scrolly = tk.Scrollbar(scroll_wrap, orient=tk.VERTICAL)
    canvas_right = tk.Canvas(scroll_wrap, bg="#1a1a1a", highlightthickness=0, yscrollcommand=scrolly.set)
    scrolly.config(command=canvas_right.yview)
    scrolly.pack(side=tk.RIGHT, fill="y")
    canvas_right.pack(side=tk.LEFT, fill="both", expand=True)
    inner_right = tk.Frame(canvas_right, bg="#1a1a1a")
    canvas_right.create_window((0, 0), window=inner_right, anchor="nw")

    def _on_inner_configure(event):
        canvas_right.configure(scrollregion=canvas_right.bbox("all"))

    inner_right.bind("<Configure>", _on_inner_configure)

    bot = tk.Label(root, text=help_txt, justify="left", font=("TkFixedFont", 9), fg="#555555")
    bot.pack(fill="x", padx=4, pady=2)

    def rebuild_roi_panels():
        nonlocal roi_panels
        for w in inner_right.winfo_children():
            w.destroy()
        roi_panels = []
        meta = state.get("meta")
        if not meta or not meta.get("rois"):
            tk.Label(inner_right, text="(no ROIs)", fg="#888", bg="#1a1a1a").pack(pady=8)
            return
        for i, roi in enumerate(meta["rois"]):
            l, t, r, b = [int(x) for x in roi["crop_box"]]
            cw, ch = r - l, b - t
            if cw <= 0 or ch <= 0:
                continue
            sc = min(roi_zoom_max_side / max(cw, ch), 10.0)
            dw = max(1, int(round(cw * sc)))
            dh = max(1, int(round(ch * sc)))
            fr = tk.Frame(inner_right, bg="#2a2a2a", relief=tk.GROOVE, bd=1)
            fr.pack(fill="x", padx=4, pady=6)
            tk.Label(
                fr,
                text=f"ROI #{i + 1}   gid={roi['group_id']}   {cw}×{ch}  px",
                fg="#ccc",
                bg="#2a2a2a",
                font=("TkFixedFont", 9),
            ).pack(anchor="w", padx=4, pady=2)
            cnv = tk.Canvas(fr, bg="#111", highlightthickness=1, highlightbackground="#444")
            cnv.pack(padx=4, pady=4)
            panel = {
                "canvas": cnv,
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "scale": sc,
                "dw": dw,
                "dh": dh,
                "photo": None,
            }
            roi_panels.append(panel)
            op, om, ore = make_roi_mouse_handlers(panel)
            cnv.bind("<ButtonPress-1>", op)
            cnv.bind("<B1-Motion>", om)
            cnv.bind("<ButtonRelease-1>", ore)

    def update_canvas():
        rebuild_roi_panels()
        refresh_all_views()

    def load_index(i):
        nonlocal idx, stroke_prev
        stroke_prev = None
        idx = i % n
        rgb = None
        gmdm_g = gwdm_g = meta = None
        for _ in range(n):
            try:
                job = image_jobs[idx]
                rgb, gmdm_g, gwdm_g, meta = load_full_image_job(job)
                break
            except (FileNotFoundError, OSError) as e:
                print(f"skip idx={idx}: {e}")
                idx = (idx + 1) % n
            except Exception as e:
                print(f"load failed idx={idx}: {e}")
                idx = (idx + 1) % n
        if rgb is None:
            print("Could not load any image after trying all items.")
            state["rgb"] = None
            state["meta"] = None
            status_var.set("Load failed")
            rebuild_roi_panels()
            return
        h, w = rgb.shape[:2]
        pores, wr = try_load_existing(meta, (h, w))
        state["rgb"] = rgb
        state["pores"] = pores
        state["wr"] = wr
        state["gmdm"] = gmdm_g
        state["gwdm"] = gwdm_g
        state["meta"] = meta
        state["scale"] = min(max_display / max(h, w), 4.0)
        update_canvas()

    def save_current():
        meta = state["meta"]
        if meta is None or state["pores"] is None:
            print("Nothing to save")
            return
        stem = stem_from_meta(meta)
        pp, wp, jp = save_paths_full_image(out_dir, stem)
        p_bin = binarize_mask_u8(state["pores"])
        w_bin = binarize_mask_u8(state["wr"])
        cv2.imwrite(pp, p_bin)
        cv2.imwrite(wp, w_bin)
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **meta,
                    "pores_gt_path": os.path.abspath(pp),
                    "wrinkles_gt_path": os.path.abspath(wp),
                    "note": "Full-size GT like gmdm/gwdm. For metrics, crop GT and pseudo by rois.crop_box then Resize(256) to match SkinDataset.mask_transform.",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved:\n{pp}\n{wp}\n{jp}")

    def on_key(e):
        nonlocal mode, brush, show_pseudo, show_rois, idx
        sym = e.keysym
        ch = (e.char or "").lower()
        if sym in ("q", "Escape"):
            root.destroy()
            return
        if sym == "n":
            load_index(idx + 1)
            return
        if sym == "b":
            load_index(idx - 1)
            return
        if ch == "p":
            mode = "pores"
            print("mode: pores")
        elif ch == "l":
            mode = "wrinkles"
            print("mode: wrinkles")
        elif ch == "e":
            mode = "erase"
            print("mode: erase")
        elif ch == "o":
            show_pseudo = not show_pseudo
            print(f"show pseudo gmdm/gwdm: {show_pseudo}")
            refresh_all_views()
        elif ch == "v":
            show_rois = not show_rois
            print(f"show ROI boxes: {show_rois}")
            refresh_all_views()
        elif sym in ("plus", "equal", "KP_Add"):
            brush = min(160, brush + 2)
            print(f"brush={brush}")
        elif sym in ("minus", "KP_Subtract"):
            brush = max(1, brush - 2)
            print(f"brush={brush}")
        elif ch == "c":
            if mode == "pores":
                state["pores"][:] = 0
            elif mode == "wrinkles":
                state["wr"][:] = 0
            else:
                state["pores"][:] = 0
                state["wr"][:] = 0
            refresh_all_views()
            print("cleared layer(s)")
        elif ch == "s":
            save_current()
        if state["meta"]:
            nroi = len(state["meta"].get("rois", []))
            status_var.set(
                f"{state['meta']['filename']}  ROIs={nroi}  [{idx + 1}/{n}]  "
                f"mode={mode}  brush={brush}  pseudo={show_pseudo}  boxes={show_rois}  |  draw on right"
            )

    def _wheel(event):
        if getattr(event, "delta", 0):
            canvas_right.yview_scroll(int(-event.delta / 120), "units")

    def _wheel_linux_up(_e):
        canvas_right.yview_scroll(-1, "units")

    def _wheel_linux_down(_e):
        canvas_right.yview_scroll(1, "units")

    canvas_right.bind("<MouseWheel>", _wheel)
    canvas_right.bind("<Button-4>", _wheel_linux_up)
    canvas_right.bind("<Button-5>", _wheel_linux_down)

    # bind_all so shortcuts work when a canvas has focus
    root.bind_all("<Key>", on_key)

    load_index(idx)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="SKIN_Q full-image labeling (GT saved like gmdm/gwdm, one file per image)")
    ap.add_argument("--config", type=str, default="../config/config_skin_Q.yaml", help="YAML with paths / ids / light_conditions")
    ap.add_argument("--json_folder", type=str, default=None)
    ap.add_argument("--image_folder", type=str, default=None)
    ap.add_argument("--split", type=str, default="train", choices=["train", "test", "full"])
    ap.add_argument("--ids", type=int, nargs="*", default=None, help="Override ids from config")
    ap.add_argument("--light_conditions", type=int, nargs="*", default=None, help="Override light_conditions from config")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="./skin_q_gt_masks")
    ap.add_argument("--max_display", type=int, default=512, help="Max side (px) for left full-image preview")
    ap.add_argument(
        "--roi_zoom",
        type=int,
        default=480,
        help="Target long side (px) for each zoomed ROI on the right",
    )
    args = ap.parse_args()

    cfg = None
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    elif not args.json_folder or not args.image_folder:
        print("Provide --config, or both --json_folder and --image_folder")
        sys.exit(1)

    ds = build_dataset(args, cfg)
    image_jobs = group_flat_items_by_image(ds)
    print(f"Unique images: {len(image_jobs)}  (flat ROI rows: {len(ds)})  use_gmdm={ds.use_gmdm} use_gwdm={ds.use_gwdm}")
    if len(ds.errors) > 0:
        print(f"JSON parse warnings/errors: {len(ds.errors)} (first 3)")
        for e in ds.errors[:3]:
            print(" ", e)

    run_ui(
        image_jobs,
        args.start,
        args.out_dir,
        max_display=args.max_display,
        roi_zoom_max_side=args.roi_zoom,
    )


if __name__ == "__main__":
    main()

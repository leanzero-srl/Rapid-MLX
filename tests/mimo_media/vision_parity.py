"""MiMo-V2.6 vision tower parity: rapid_mlx (MLX) vs the HF reference (torch, CPU).

Two interpreters, one file:
  1. `<mlx python> vision_parity.py prep  --model-dir D --out X.npz`   images -> patches (ours)
  2. `<torch python> vision_parity.py ref --model-dir D --weights W --in X.npz --out R.npz`
       HF Qwen2VLImageProcessor patches (if transformers is there) + reference forward,
       every layer captured in row order, float32
  3. `<mlx python> vision_parity.py cmp   --model-dir D --weights W --in X.npz --ref R.npz`
       our forward (float32 and bfloat16), cosine per layer + merger, and our patches vs HF's
The images: a rendered table (the Gauntlet case: text in cells), a non-square gradient
(column-ordered window blocks and the 2-D rope both see h != w), and noise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def images():
    from PIL import Image, ImageDraw

    t = Image.new("RGB", (256, 160), "white")
    d = ImageDraw.Draw(t)
    for r in range(5):
        for c in range(3):
            d.rectangle([c * 85, r * 32, c * 85 + 84, r * 32 + 31], outline="black")
            d.text((c * 85 + 6, r * 32 + 10), f"R{r}C{c}={r * 7 + c}", fill="black")
    g = np.zeros((96, 352, 3), np.uint8)
    g[..., 0] = np.linspace(0, 255, 352)[None, :]
    g[..., 1] = np.linspace(0, 255, 96)[:, None]
    g[..., 2] = 128
    rng = np.random.default_rng(7)
    n = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    # a screenshot-sized page (~1.9k patches: many band blocks, several 64-block groups)
    page = Image.new("RGB", (800, 600), (240, 244, 250))
    d = ImageDraw.Draw(page)
    d.rectangle([0, 0, 800, 48], fill=(30, 90, 200))
    d.text((16, 16), "Orders - Gauntlet selftest", fill="white")
    for r in range(14):
        y = 70 + r * 36
        d.rectangle([20, y, 780, y + 32], outline=(120, 120, 120), fill="white" if r % 2 else (230, 236, 245))
        d.text((30, y + 10), f"#{1000 + r}   customer {chr(65 + r)}   total {r * 13.5:.2f} EUR", fill="black")
    return {"table": t, "gradient": Image.fromarray(g), "noise": Image.fromarray(n),
            "page": page}


def prep(a):
    from rapid_mlx.mlx_lm_models import mimo_v2_media as M

    c = M.MediaConfig.from_model_dir(a.model_dir)
    out = {}
    for name, img in images().items():
        p, grid = M.image_patches(img, c)
        out[f"{name}_patches"] = p
        out[f"{name}_grid"] = np.array(grid)
        img.save(Path(a.out).with_name(f"{name}.png"))
    np.savez(a.out, **out)
    print("prep", {k: v.shape for k, v in out.items() if k.endswith("patches")})


def ref(a):
    import torch
    from safetensors.torch import load_file
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parent))
    import hf_vision_reference as R

    cfg = json.loads((Path(a.model_dir) / "config.json").read_text())
    model = R.MiMoVisionTransformer(SimpleNamespace(**cfg["vision_config"])).float().eval()
    sd = {k[len("vision_tower."):]: v.float() for k, v in load_file(a.weights).items()
          if k.startswith("vision_tower.")}
    pe = sd["patch_embed.proj.weight"]
    if pe.ndim == 2:  # mlx-vlm layout (out, C*T*ps*ps) -> Conv3d (out, C, T, ps, ps)
        sd["patch_embed.proj.weight"] = pe.reshape(pe.shape[0], 3, 2, 16, 16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the checkpoint has no merger biases (mlx-vlm builds bias-free); zero the torch ones
    for name in missing:
        assert name.endswith(".bias") and "merger" in name, missing
        dict(model.named_parameters())[name].data.zero_()
    assert not unexpected, unexpected
    data = np.load(a.inp)
    out = {}
    hf = None
    try:
        from transformers import Qwen2VLImageProcessor
        # Xiaomi's values (config.json processor_config + upstream preprocessor_config.json),
        # which MediaConfig uses, not a converted build's preprocessor_config.json.
        pc = cfg.get("processor_config") or {}
        hf = Qwen2VLImageProcessor(
            min_pixels=pc.get("image_min_pixels", 8192), max_pixels=pc.get("image_max_pixels", 8388608),
            patch_size=16, temporal_patch_size=2, merge_size=2,
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
    except Exception as e:  # noqa: BLE001
        print("no HF image processor:", e)
    from PIL import Image
    for name in images():
        x = torch.from_numpy(data[f"{name}_patches"]).float()
        grid = torch.from_numpy(data[f"{name}_grid"]).long()[None]
        if hf is not None:
            r = hf(images=[Image.open(Path(a.inp).with_name(f"{name}.png"))], return_tensors="np")
            out[f"{name}_hfpatches"] = r["pixel_values"]
            out[f"{name}_hfgrid"] = r["image_grid_thw"][0]
        layers = []
        with torch.no_grad():
            m = model
            h = m.patch_embed(x)
            rot = m.rot_pos_emb(grid)
            emb = torch.cat((rot, rot), dim=-1)
            wi = m.get_window_index_1d(grid, col=True)
            rev = torch.argsort(wi)
            rowe = (emb.cos(), emb.sin())
            ce = m.apply_index(emb, wi)
            cole = (ce.cos(), ce.sin())
            cu = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0]).cumsum(0, dtype=torch.int32)
            cu = torch.nn.functional.pad(cu, (1, 0))
            T = m.vit_window_attn_types
            for i, blk in enumerate(m.blocks):
                if T[i] == 1 and (i == 0 or T[i - 1] != 1):
                    h = m.apply_index(h, wi)
                if i > 0 and T[i] != 1 and T[i - 1] == 1:
                    h = m.apply_index(h, rev)
                pe_ = cole if T[i] == 1 else rowe
                h = blk(h, cu_seqlens=cu, position_embeddings=pe_, full_attn=i in m.fullatt_block_indexes)
                layers.append((m.apply_index(h, rev) if T[i] == 1 else h).numpy())
            final = m.merger(h).numpy()
            full = m(x, grid).numpy()
        assert np.allclose(final, full, atol=1e-4), "capture loop != reference forward"
        with torch.no_grad():  # the reference itself in bf16: the floor for our bf16 run
            out[f"{name}_final_bf16"] = model.to(torch.bfloat16)(x.bfloat16(), grid).float().numpy()
            model.float()
        out[f"{name}_layers"] = np.stack(layers)
        out[f"{name}_final"] = final
    np.savez(a.out, **out)
    print("ref done")


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def cmp(a):
    import mlx.core as mx
    from rapid_mlx.mlx_lm_models import mimo_v2_media as M

    data, r = np.load(a.inp), np.load(a.ref)
    report = {}
    for dtype in (mx.float32, mx.bfloat16):
        model, _ = M.load_vision(a.model_dir, dtype=dtype, weights_file=a.weights)
        for name in images():
            x = mx.array(data[f"{name}_patches"])
            grid = [tuple(int(v) for v in data[f"{name}_grid"])]
            final, layers = model(x, grid, return_layers=True)
            per = [cos(np.array(l.astype(mx.float32)), r[f"{name}_layers"][i]) for i, l in enumerate(layers)]
            fin = cos(np.array(final.astype(mx.float32)), r[f"{name}_final"])
            ref_bf16 = cos(r[f"{name}_final_bf16"], r[f"{name}_final"])
            report[f"{name}/{'f32' if dtype == mx.float32 else 'bf16'}"] = {
                "tokens": int(final.shape[0]), "min_layer_cos": round(min(per), 6),
                "worst_layer": int(np.argmin(per)), "final_cos": round(fin, 6),
                "torch_bf16_final_cos": round(ref_bf16, 6)}
    for name in images():
        if f"{name}_hfpatches" in r:
            ours, hf = data[f"{name}_patches"], r[f"{name}_hfpatches"]
            same_grid = list(data[f"{name}_grid"]) == list(r[f"{name}_hfgrid"])
            report[f"{name}/patches_vs_hf"] = {
                "grid_equal": same_grid,
                "max_abs": float(np.abs(ours - hf).max()) if ours.shape == hf.shape else None}
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "ref", "cmp"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"prep": prep, "ref": ref, "cmp": cmp}[a.mode](a)

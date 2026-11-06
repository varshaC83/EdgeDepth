"""Generate all result tables in the docs FROM the JSON files written by the
evaluation scripts, so no number in the documentation is typed by hand.

    python scripts/make_report.py

Writes:
  * eval_P4_TeamID.md                (official metric order)
  * docs/results_tables.md
  * replaces the blocks between <!-- AUTO:name --> and <!-- /AUTO:name --> in
    README.md, docs/design_document.md, docs/ablation_study.md, docs/failure_analysis.md
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth import METHOD_LABELS  # noqa: E402

R = "results"
M7 = ["nearest", "bilinear", "bicubic", "joint_bilateral", "guided_filter", "deep_learning", "hybrid"]


def load(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None


def f(v, d=4):
    return "–" if v is None else f"{v:.{d}f}"


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def table(rows, cols, header, bold_best=None, lower_better=()):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    best = {}
    if bold_best:
        for c in bold_best:
            vals = [r[c] for r in rows if r.get(c) is not None]
            if vals:
                best[c] = min(vals) if c in lower_better else max(vals)
    for r in rows:
        cells = []
        for c, d in cols:
            if callable(c):
                cells.append(c(r))
                continue
            s = f(r.get(c), d)
            if c in best and r.get(c) is not None and abs(r[c] - best[c]) < 1e-12:
                s = f"**{s}**"
            cells.append(s)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


LOWER = ("rmse", "mae", "b_rmse", "latency_ms")
MAIN_COLS = [(lambda r: METHOD_LABELS[r["method"]], 0), ("rmse", 4), ("mae", 4), ("psnr", 2), ("ssim", 4), ("b_rmse", 4),
             ("edge_f1", 3), ("edge_sharpness", 3), ("latency_ms", 0)]
MAIN_HDR = ["Method", "RMSE (m) ↓", "MAE (m) ↓", "PSNR (dB) ↑", "SSIM ↑", "Boundary RMSE (m) ↓", "Edge F1 ↑", "Edge sharpness (→1)", "Latency (ms) ↓"]


def ordered(rows):
    return sorted(rows, key=lambda r: M7.index(r["method"]) if r["method"] in M7 else 99)


def main_tables(s):
    parts = {}
    rows = ordered(s["overall"])
    parts["overall"] = table(rows, MAIN_COLS, MAIN_HDR, ["rmse", "mae", "psnr", "ssim", "b_rmse", "edge_f1"], LOWER)
    blocks = []
    for sc in ("2x", "4x", "8x"):
        for cond, title in (("clean", "clean"), ("noisy_missing", "noisy + missing")):
            rr = ordered(pick(s["by_scale_condition"], scale=sc, condition=cond))
            if rr:
                blocks.append(f"**{sc}, {title}** (n = {rr[0]['n']} frames)\n\n" + table(rr, MAIN_COLS, MAIN_HDR, ["rmse", "mae", "psnr", "ssim", "b_rmse", "edge_f1"], LOWER))
    rr = ordered(pick(s["by_scale_condition"], condition="sparse_5pct"))
    if rr:
        blocks.append(f"**Sparse input, 5 % of pixels** (n = {rr[0]['n']} frames)\n\n" + table(rr, MAIN_COLS, MAIN_HDR, ["rmse", "mae", "psnr", "ssim", "b_rmse", "edge_f1"], LOWER))
    parts["detailed"] = "\n\n".join(blocks)
    # per scale summary (mean over clean & noisy)
    hdr = ["Method"] + [f"RMSE {s_}" for s_ in ("2x", "4x", "8x")] + [f"B-RMSE {s_}" for s_ in ("2x", "4x", "8x")] + [f"Edge F1 {s_}" for s_ in ("2x", "4x", "8x")]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for m in M7:
        r = {sc: next((x for x in s["by_scale"] if x["scale"] == sc and x["method"] == m), None) for sc in ("2x", "4x", "8x")}
        if not all(r.values()):
            continue
        lines.append("| " + " | ".join([METHOD_LABELS[m]] + [f(r[sc]["rmse"]) for sc in r] + [f(r[sc]["b_rmse"]) for sc in r] + [f(r[sc]["edge_f1"], 3) for sc in r]) + " |")
    parts["by_scale"] = "\n".join(lines)
    mem = s.get("memory_mb_4x", {})
    lat4 = {r["method"]: r["latency_ms"] for r in s["by_scale"] if r["scale"] == "4x"}
    lines = ["| Method | Mean latency at 4× (ms) | Peak memory at 4× (MB, approx.) |", "|---|---|---|"]
    for m in M7:
        if m in mem:
            lines.append(f"| {METHOD_LABELS[m]} | {f(lat4.get(m), 0)} | {f(mem[m], 1)} |")
    parts["efficiency"] = "\n".join(lines)
    cd = ordered(pick(s["cross_dataset"], scale="8x")) if s.get("cross_dataset") else []
    if cd:
        parts["cross"] = "**8× upsampling, mean over TUM RGB-D / SUN RGB-D / Redwood frames**\n\n" + table(
            cd, MAIN_COLS[:8], MAIN_HDR[:8], ["rmse", "mae", "b_rmse", "edge_f1"], LOWER)
        cd4 = ordered(pick(s["cross_dataset"], scale="4x"))
        parts["cross"] += "\n\n**4× upsampling**\n\n" + table(cd4, MAIN_COLS[:8], MAIN_HDR[:8], ["rmse", "mae", "b_rmse", "edge_f1"], LOWER)
    return parts


def ablation_table(a):
    rows = [r for r in a["rows"] if r["scale"] == "all"]
    cols = [(lambda r: r["variant"], 0), ("rmse", 4), ("mae", 4), ("psnr", 2), ("ssim", 4), ("b_rmse", 4), ("edge_f1", 3), ("edge_sharpness", 3), ("latency_ms", 0)]
    hdr = ["Variant", "RMSE ↓", "MAE ↓", "PSNR ↑", "SSIM ↑", "B-RMSE ↓", "Edge F1 ↑", "Sharpness", "Latency (ms)"]
    t = table(rows, cols, hdr, ["rmse", "mae", "psnr", "ssim", "b_rmse", "edge_f1"], LOWER)
    blocks = [t]
    for sc in ("2x", "4x", "8x"):
        rr = [r for r in a["rows"] if r["scale"] == sc]
        blocks.append(f"**{sc}** (clean + noisy/missing)\n\n" + table(rr, cols[:7], hdr[:7], ["rmse", "mae", "psnr", "ssim", "b_rmse", "edge_f1"], LOWER))
    return "\n\n".join(blocks)


def failure_tables(fc):
    M = ["bicubic", "joint_bilateral", "guided_filter", "deep_learning", "hybrid"]
    M = [m for m in M if m in fc["F1_sparsity"]["0.1"]]
    out = {}
    dens = list(fc["F1_sparsity"])
    lines = ["| Method | " + " | ".join(f"{float(d) * 100:g} %" for d in dens) + " |", "|---|" + "---|" * len(dens)]
    for m in M:
        lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(f(fc["F1_sparsity"][d][m]["rmse"]) for d in dens) + " |")
    out["F1"] = "RMSE (m) vs sparse density\n\n" + "\n".join(lines)
    lines = ["| Method | RMSE inside hole (m) | RMSE outside hole (m) |", "|---|---|---|"]
    for m in M:
        v = fc["F2_large_hole_rmse"][m]
        lines.append(f"| {METHOD_LABELS[m]} | {f(v['inside'])} | {f(v['outside'])} |")
    out["F2"] = "\n".join(lines)
    nl = list(fc["F3_noise"])
    lines = ["| Method | " + " | ".join(f"noise ×{n}" for n in nl) + " |", "|---|" + "---|" * len(nl)]
    for m in M:
        lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(f(fc["F3_noise"][n][m]["rmse"]) for n in nl) + " |")
    out["F3"] = "RMSE (m) at 4×\n\n" + "\n".join(lines)
    lines = ["| Method | RMSE (mm) | MAE on RGB texture edges (mm) | MAE elsewhere (mm) | Texture-copy ratio (1 = none) |", "|---|---|---|---|---|"]
    for m in M:
        v = fc["F4_texture_copy"][m]
        lines.append(f"| {METHOD_LABELS[m]} | {f(v['rmse'] * 1000, 2)} | {f(v['mae_on_rgb_edges_mm'], 2)} | {f(v['mae_off_rgb_edges_mm'], 2)} | {f(v['texture_copy_ratio'], 2)} |")
    out["F4"] = "\n".join(lines)
    sh = list(fc["F5_misalignment"])
    lines = ["| Method | " + " | ".join(f"shift {s_} px" for s_ in sh) + " |", "|---|" + "---|" * len(sh)]
    for m in M:
        lines.append(f"| {METHOD_LABELS[m]} | " + " | ".join(f"{f(fc['F5_misalignment'][s_][m]['b_rmse'])} / {f(fc['F5_misalignment'][s_][m]['edge_f1'], 3)}" for s_ in sh) + " |")
    out["F5"] = "Boundary RMSE (m) / Edge F1 at 8×\n\n" + "\n".join(lines)
    return out


def replace_blocks(path, blocks):
    if not os.path.exists(path):
        return
    txt = open(path, encoding="utf-8").read()
    for name, content in blocks.items():
        pat = re.compile(rf"(<!-- AUTO:{re.escape(name)} -->)(.*?)(<!-- /AUTO:{re.escape(name)} -->)", re.S)
        txt = pat.sub(lambda m: m.group(1) + "\n" + content + "\n" + m.group(3), txt)
    open(path, "w", encoding="utf-8").write(txt)


def eval_file(s, parts, a, fc, tl, ck):
    hy = next(r for r in s["overall"] if r["method"] == "hybrid")
    bc = next(r for r in s["overall"] if r["method"] == "bicubic")
    L = ["<USN_1>,<USN_2>,<USN_3>,<USN_4>", "",
         "# Evaluation — P4: Edge-Preserving Depth Map Upsampling and Refinement (Team <TeamID>)", "",
         "Primary method: **Hybrid** (cue-driven fusion of bicubic, guided filter, JBU and EdgeDepthNet + noise-adaptive "
         "edge-aware back-projection). All numbers below were produced by `scripts/evaluate.py`, "
         "`scripts/ablation.py` and `scripts/failure_cases.py` on the unseen, scene-disjoint NYU Depth V2 test split "
         f"({len(s['protocol']['test_frames'])} frames, {s['protocol']['resolution'][1]}×{s['protocol']['resolution'][0]}), "
         f"on {s['protocol']['machine']}. Regenerate with `python scripts/make_report.py`.", "",
         "Conditions: *clean* = box-downsampled GT; *noisy + missing* = Kinect noise model ×1, 5 % random dropout, "
         "50 % dropout on depth edges; *sparse* = 5 % random samples. Averages are over all frames and conditions unless stated.", "",
         "## Headline (hybrid vs bicubic, mean over 2×/4×/8× and all conditions)", "",
         "| Metric | Bicubic | Hybrid | Change |", "|---|---|---|---|"]
    for k, lab, d in (("rmse", "RMSE (m)", 4), ("mae", "MAE (m)", 4), ("psnr", "PSNR (dB)", 2), ("ssim", "SSIM", 4),
                      ("b_rmse", "Boundary RMSE (m)", 4), ("edge_f1", "Edge F1", 3)):
        ch = (hy[k] - bc[k]) / abs(bc[k]) * 100
        L.append(f"| {lab} | {f(bc[k], d)} | {f(hy[k], d)} | {ch:+.1f} % |")
    L += ["", "## 1. RMSE and MAE (depth accuracy)", "", "Mean over all test frames, scales and conditions:", "",
          table(ordered(s["overall"]), MAIN_COLS[:3], MAIN_HDR[:3], ["rmse", "mae"], LOWER), "",
          "## 2. PSNR and SSIM", "", "PSNR peak = 10 m; SSIM data range = per-frame GT range.", "",
          table(ordered(s["overall"]), [MAIN_COLS[0], MAIN_COLS[3], MAIN_COLS[4]], [MAIN_HDR[0], MAIN_HDR[3], MAIN_HDR[4]], ["psnr", "ssim"], LOWER), "",
          "## 3. Edge preservation and boundary accuracy", "",
          "Boundary RMSE = RMSE within 2 px of GT depth discontinuities; Edge F1 = F1 of discontinuity maps (2-px tolerance); "
          "edge sharpness = mean |∇log D| on GT edges relative to GT (1.0 ideal).", "",
          table(ordered(s["overall"]), [MAIN_COLS[0]] + MAIN_COLS[5:8], [MAIN_HDR[0]] + MAIN_HDR[5:8], ["b_rmse", "edge_f1"], LOWER), "",
          "## 4. Performance across upsampling factors", "", "Mean of clean and noisy + missing conditions:", "", parts["by_scale"], "",
          "Full per-condition tables:", "", parts["detailed"], "",
          "## 5. Processing latency and memory consumption", "", parts["efficiency"], "",
          "Latency is wall-clock on the CPU above (no GPU); memory = max(tracemalloc peak, sampled RSS increase).", ""]
    if ck:
        L += [f"EdgeDepthNet: {ck.get('params', 0):,} parameters, best checkpoint at iteration {ck.get('iter')} "
              f"(val mean RMSE {ck.get('val_rmse', {}).get('mean', float('nan')):.4f} m), CPU training time {ck.get('train_time_min')} min.", ""]
    if parts.get("cross"):
        L += ["## Additional: cross-dataset generalisation (unseen sensors)", "", parts["cross"], ""]
    if a:
        L += ["## Additional: ablation (test split)", "", ablation_table(a).split("\n\n")[0], ""]
    return "\n".join(L)


def main():
    s, a, fc, tl = load("summary.json"), load("ablation.json"), load("failure_cases.json"), load("training_log.json")
    ck = None
    try:
        from edgedepth.model.inference import model_info

        ck = model_info()
    except Exception:
        pass
    if not s:
        sys.exit("results/summary.json missing — run scripts/evaluate.py first")
    parts = main_tables(s)
    blocks = {"overall": parts["overall"], "by_scale": parts["by_scale"], "detailed": parts["detailed"],
              "efficiency": parts["efficiency"], "cross": parts.get("cross", "")}
    if a:
        blocks["ablation"] = ablation_table(a)
    if fc:
        for k, v in failure_tables(fc).items():
            blocks[f"failure_{k}"] = v
    with open("docs/results_tables.md", "w", encoding="utf-8") as fh:
        fh.write("# Auto-generated result tables\n\nGenerated by `scripts/make_report.py` from `results/*.json`.\n\n")
        for k, v in blocks.items():
            fh.write(f"## {k}\n\n{v}\n\n")
    for p in ("README.md", "docs/design_document.md", "docs/ablation_study.md", "docs/failure_analysis.md"):
        replace_blocks(p, blocks)
    with open("eval_P4_TeamID.md", "w", encoding="utf-8") as fh:
        fh.write(eval_file(s, parts, a, fc, tl, ck))
    print("report written")


if __name__ == "__main__":
    main()

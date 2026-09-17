import os, json, time, numpy as np
import harness_fe as H
import features_candidates as C

df = H.load_base()
base_cols = H._base_temporal_cols(df)

CANDS = {
    "1_multiwindow":   C.add_multiwindow,
    "2_flips":         C.add_flips,
    "3_xs_temporal":   C.add_xs_temporal,
    "4_close_interac": C.add_close_interactions,
}

# base CV (cache)
if os.path.exists("scratch_base_cv.json"):
    base_cv = json.load(open("scratch_base_cv.json"))
else:
    base_cv = H.cv_one(df, base_cols); json.dump(base_cv, open("scratch_base_cv.json", "w"))
base_res, _ = H.blend_gate(df, base_cols, [])
print("BASE CV per-fold", [round(x,4) for x in base_cv], "mean", round(np.mean(base_cv),4), flush=True)
print("BASE blend", {k:round(v,4) for k,v in base_res.items()}, flush=True)

results = {"base": {"cv": base_cv, "blend": base_res}}
for name, fn in CANDS.items():
    t = time.time()
    d = df.copy()
    new = fn(d)
    d[new] = d[new].replace([np.inf, -np.inf], np.nan)
    cv = H.cv_one(d, base_cols + new)
    bl, _ = H.blend_gate(d, base_cols, new)
    results[name] = {"new_cols": new, "cv": cv, "blend": bl}
    print(f"\n=== {name} ({len(new)} feats, {round(time.time()-t)}s) ===", flush=True)
    print("  CV per-fold", [round(x,4) for x in cv], "mean", round(np.mean(cv),4),
          "delta", round(np.mean(cv)-np.mean(base_cv),4), flush=True)
    print("  blend", {k:round(v,4) for k,v in bl.items()},
          "d_blend", round(bl["blend_skill"]-base_res["blend_skill"],4), flush=True)
    json.dump(results, open("scratch_results.json", "w"), default=float)

print("\nDONE", flush=True)

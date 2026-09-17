# Optiver — Trading at the Close

Practice project on the Kaggle competition [Optiver – Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close).
The goal here was to learn the full applied-ML arc end to end — EDA → validation →
modelling → ensembling → shipping through a real code-competition pipeline — with
every decision backed by a measurement.

**Final private leaderboard score: 5.4777 MAE.**

| Submission | Private score |
|---|---|
| LightGBM only | 5.5 |
| LGB + GRU blend | 5.4896 |
| **LGB + GRU + Transformer blend** | **5.4777** |

---

## The problem

During the last 10 minutes of Nasdaq trading, predict each stock's **60-second future
WAP move relative to a synthetic index**, in basis points. Scored on **MAE**. It is a
*code competition*: submissions must run inside a Kaggle notebook against a hidden test
set served one time-bucket at a time by a time-series API, so the model physically
cannot see the future.

**Data:** 5.24M rows — 200 stocks × 481 dates × 55 timesteps (seconds 0…540, every 10s).
The target is ~zero-centred and fat-tailed; predicting all zeros gives MAE ≈ 6.4.

---

## Repository layout

| File | Purpose |
|---|---|
| `features.py` | Leakage-safe feature engineering (single source of truth) |
| `EDA.ipynb` | Exploratory analysis — target, imbalance, price relationships |
| `Baseline.ipynb` | LightGBM baseline, walk-forward CV harness, blending |
| `LSTM.ipynb` | GRU / LSTM sequence models over the 55-step auction |
| `transformer.ipynb` | Causal Transformer (the model that broke the ceiling) |
| `submission.ipynb` | Inference against the time-series API |
| `infer_fast.py` | Stateful inference: `StatefulGRU` (carried hidden state), `KVCacheTransformer` (per-layer KV cache) |
| `bench_inference.py` | Profiling harness — per-component timing, within-day cost growth, identical-prediction check |
| `artifacts/` | Trained models in **portable** formats + `meta.json` |

### Feature builders

- `build_features(df)` → **33 features**: static price/size combinations plus
  *cross-sectional* (peer-relative) rank/z at each instant. Used by the sequence models.
- `build_features_gbdt(df)` → **44 features**: the above plus **within-day temporal**
  features (diffs, 60s momentum, rolling vol). Used by LightGBM — trees can't derive
  temporal structure, recurrent nets can.

Every feature uses only the same row, same-instant peers, or *strictly earlier* steps.

---

## Final model

```
0.3 × LightGBM(44 temporal feats)
0.1 × GRU        (3 seeds averaged)
0.6 × Transformer(3 seeds averaged)
   → subtract the per-(date, second) mean   [zero-sum post-processing]
```

Both neural nets read the 55-step sequence and are **strictly causal**
(`bidirectional=False` for the RNN; an upper-triangular attention mask for the
Transformer). This is the leakage guard — and it means a partial sequence `[0…t]`
produces exactly the same output at step `t` as the full sequence, which is what makes
streaming inference correct.

---

## Methodology (the part worth reusing)

1. **Never split randomly.** Split by `date_id`; validate on later dates only.
2. **Report *skill %***, not raw MAE: `(zero_MAE − model_MAE) / zero_MAE × 100`.
   Raw MAE isn't comparable across periods with different volatility — a calm
   validation window flatters a model for free.
3. **Walk-forward CV** (multiple sequential folds) to establish a **noise floor**.
   Ours was ±0.09% skill — anything smaller than that is not a result.
4. **Paired vs unpaired comparisons.** A tweak applied to identical predictions
   (e.g. post-processing) is judged by *directional consistency across folds*; a
   different model is judged by *margin over the noise floor*.
5. **Ensembles need decorrelation, not stronger parts.** A better single model does
   **not** help the blend if it becomes more correlated with the others.

---

## What worked, and what didn't

| Attempt | Result |
|---|---|
| LightGBM + engineered features | skill ≈ 2.0% — plateaued |
| Heavier regularisation / tuning | ✗ no change (not an overfitting problem) |
| Cross-day *historical* features | ✗ nothing |
| Market-wide *global* features | ✗ nothing |
| **Within-day *temporal* features** | ✓ base 1.78% → 2.01% (only feature family that paid) |
| GRU (multi-seed) | ✓ decorrelated from LGB (corr 0.86) → blend gain |
| **LSTM added to the blend** | ✗ corr 0.957 with the GRU — optimal weight was **zero** |
| **Transformer added to the blend** | ✓ corr 0.892, best single model → **blend 2.22% → 2.40%** |
| Zero-sum post-processing | ~ tiny, free, kept |
| 2nd feature pass (multi-window, accel/flips, xsec-of-temporal, close interactions) | ✗ max +0.006% vs a 0.09% noise floor — **tapped out** (verified by two independent runs) |

The two ✗ results that taught the most: the **LSTM** (too similar to the GRU to add
anything) and the **second feature pass** (the neural nets already learn those patterns,
so hand-crafted versions add nothing to the *ensemble*).

---

## Reproducing

```bash
# 1. Features + LightGBM baseline + walk-forward CV
jupyter lab Baseline.ipynb

# 2. Sequence models (saves per-seed weights into artifacts/)
jupyter lab LSTM.ipynb          # GRU
jupyter lab transformer.ipynb   # causal Transformer

# 3. Blend weights are searched on a held-out 20% of dates, then
#    inference runs through the API in submission.ipynb
```

### Submitting (code competition)

1. Upload `artifacts/` as a Kaggle **Dataset** (portable formats only —
   `lgb_final.txt`, `*.pt` state-dicts, `scaler.npz`, `meta.json`).
2. Work from a notebook **forked from a 2023-era notebook** so the environment pins
   **Python 3.10** — the competition's `optiver2023` API ships a `cpython-310` binary
   and will not import on a newer runtime.
3. Save & Run All (Commit) → Submit the notebook version.

```bash
kaggle competitions submit optiver-trading-at-the-close \
  -k <user>/<notebook-slug> -v <version> -f submission.csv -m "message"
```

---

## Environment gotchas

- **Never pickle models across environments.** `joblib` of a LightGBM object breaks on a
  different LightGBM version (`'Booster' object has no attribute 'handle'`). Use native
  formats: `booster.save_model(...)` and `torch.save(state_dict)`.
- **macOS only:** importing both `lightgbm` and `torch` in one process segfaults
  (duplicate OpenMP runtimes). Prefix with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`.
  Does not occur on Linux/Kaggle.
- **Conda `pytorch` from the `defaults` channel is broken on Apple Silicon**
  (libtorch symbol mismatch). Install the PyPI wheel: `pip install torch`.

---

## Performance engineering (inference)

The first submission took **over an hour**. Profiling the loop with `bench_inference.py`
(on the 3 example days, 165 buckets) showed the bottleneck was *not* where intuition
pointed:

| Component | Share of time |
|---|---|
| Transformer forward ×3 seeds | **63%** |
| GRU forward ×3 seeds | 18% |
| Feature rebuild on the day buffer | 14% |
| LightGBM | 3% |

The nets were being re-run on the **full sequence `[0…t]` at every bucket** just to read
the output at step `t` — O(T) for the GRU and O(T²) for the Transformer's attention —
so the last bucket of each day cost 7.5× the first. (The feature-rebuild "fix" that
seemed obvious gave a **1.0×** speedup: wrong target. Measure first.)

**Fix: stateful inference.** Both nets are strictly causal, so their internal state for
steps `0…t−1` is identical at bucket `t` to what was computed at bucket `t−1`. Keep it
instead of recomputing it (`infer_fast.py`, trained weights untouched):

- `StatefulGRU` — carry the hidden state `h`, feed only the new step. GRU **23×** faster.
- `KVCacheTransformer` — cache each layer's keys/values, attend one new query over the
  cache (the same trick LLM text generation uses). Transformer **6.9×** faster; the
  causal mask disappears because the cache only ever contains the past.

| | ms / bucket | within-day growth | max \|Δprediction\| |
|---|---|---|---|
| baseline | 199 | 7.5× | — |
| + stateful GRU | 167 | 5.9× | 2.4e-07 |
| **+ KV-cache Transformer** | **56** | **1.6×** | 1.9e-06 |

**3.5× overall, predictions identical to float32 noise.** Two lessons worth keeping:
*Amdahl's law* (a component can only buy you speedup in proportion to its share — the
GRU at 18% capped out near 1.2×) and *the bottleneck moves* (after the fix, the feature
rebuild became 50% of the time). The remaining obvious step — trim the raw buffer to
the last ~8 buckets, since temporal windows reach back only 6 — was left undone when the
project was closed.

The only untapped modelling lever is **online learning**: the hidden test spans months
after the training data, and the API streams `revealed_targets` (previous day's true
targets) precisely so models can adapt to drift. That is what separated the top of the
leaderboard.

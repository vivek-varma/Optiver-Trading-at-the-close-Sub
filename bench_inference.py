"""Profile the submission inference loop, component by component.

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python bench_inference.py

Reports (a) total time per component and (b) per-bucket cost vs bucket index,
which exposes the quadratic growth from rebuilding features on the whole day.
"""
import json, time, numpy as np, pandas as pd, torch, torch.nn as nn, lightgbm as lgb
from collections import defaultdict
import features as F
from infer_fast import StatefulGRU, KVCacheTransformer

torch.set_num_threads(1)
ART = "artifacts"


class SeqModel(nn.Module):
    def __init__(self, nf, ns, emb=24, h=128, l=2, dr=0.2):
        super().__init__()
        self.emb = nn.Embedding(ns, emb)
        self.rnn = nn.GRU(nf + emb, h, l, batch_first=True, dropout=dr, bidirectional=False)
        self.head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, st):
        e = self.emb(st).unsqueeze(1).expand(-1, x.size(1), -1)
        o, _ = self.rnn(torch.cat([x, e], -1))
        return self.head(o).squeeze(-1)


class TimeTransformer(nn.Module):
    def __init__(self, nf, ns, emb_dim=24, d_model=128, nhead=4, layers=3, dropout=0.2, seq_len=55):
        super().__init__()
        self.emb = nn.Embedding(ns, emb_dim)
        self.in_proj = nn.Linear(nf + emb_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        lyr = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(lyr, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, st):
        T = x.size(1)
        e = self.emb(st).unsqueeze(1).expand(-1, T, -1)
        h = self.in_proj(torch.cat([x, e], -1)) + self.pos[:, :T]
        cm = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        return self.head(self.encoder(h, mask=cm)).squeeze(-1)


def load_models():
    meta = json.load(open(f"{ART}/meta.json"))
    sc = np.load(f"{ART}/scaler.npz")
    booster = lgb.Booster(model_file=f"{ART}/lgb_final.txt")
    NF, NS = meta["n_features"], meta["n_stocks"]
    grus = [SeqModel(NF, NS).eval() for _ in range(3)]
    for i, g in enumerate(grus):
        g.load_state_dict(torch.load(f"{ART}/gru_seed{i}.pt", map_location="cpu"))
    tfms = [TimeTransformer(NF, NS, **meta["transformer_arch"]).eval() for _ in range(3)]
    for i, t in enumerate(tfms):
        t.load_state_dict(torch.load(f"{ART}/transformer_seed{i}.pt", map_location="cpu"))
    return meta, sc["mean"], sc["scale"], booster, grus, tfms


def run(tail=None, profile=True, stateful_gru=False, stateful_tfm=False):
    """tail=None -> rebuild features on the whole day (current submission code).
       tail=N    -> only keep the last N buckets of raw rows.
       stateful_gru=True -> carry the GRU hidden state; feed only the new step.
       stateful_tfm=True -> KV-cache the Transformer; feed only the new step."""
    meta, MEAN, SCALE, booster, grus, tfms = load_models()
    feat_cols, gcols = meta["feat_cols"], meta["gbdt_cols"]
    stock_to_idx = {int(k): v for k, v in meta["stock_to_idx"].items()}
    WL, WG, WT = 0.3, 0.1, 0.6
    sgrus = [StatefulGRU(g) for g in grus] if stateful_gru else None
    stfms = [KVCacheTransformer(m) for m in tfms] if stateful_tfm else None
    need_seq = not (stateful_gru and stateful_tfm)   # full sequences only if someone still needs them

    test_all = pd.read_csv("Data/example_test_files/test.csv")
    samp_all = pd.read_csv("Data/example_test_files/sample_submission.csv")

    t = defaultdict(float)          # component -> seconds
    per_bucket = []                 # (bucket_idx_in_day, seconds)
    preds = []
    cur_day, raw, hist, step_in_day = None, [], defaultdict(list), 0
    t_start = time.perf_counter()

    for tid in test_all.time_id.drop_duplicates():
        b0 = time.perf_counter()
        test = test_all[test_all.time_id == tid].copy()
        sample = samp_all[samp_all.time_id == tid].copy()
        d = int(test.date_id.iloc[0]); sec = int(test.seconds_in_bucket.iloc[0])
        if d != cur_day:
            cur_day, raw, hist, step_in_day = d, [], defaultdict(list), 0
            for s in (sgrus or []) + (stfms or []):
                s.reset()                         # new day = new sequence = blank memory
        raw.append(test)
        if tail:
            raw = raw[-tail:]

        a = time.perf_counter()
        day = F.build_features_gbdt(pd.concat(raw, ignore_index=True))
        cur = day[day.seconds_in_bucket == sec]
        t["1_features"] += time.perf_counter() - a

        stocks = cur.stock_id.to_numpy()

        a = time.perf_counter()
        p_lgb = booster.predict(cur[gcols + ["stock_id"]].replace([np.inf, -np.inf], np.nan).fillna(0))
        t["2_lgb"] += time.perf_counter() - a

        a = time.perf_counter()
        base = ((cur[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy() - MEAN) / SCALE).astype(np.float32)
        x_t = torch.tensor(base)                                  # (n_stocks, 33): the new step
        sidx = torch.tensor([stock_to_idx.get(int(s), 0) for s in stocks], dtype=torch.long)
        seq = None
        if need_seq:                                              # only the full re-forward paths need [0..t]
            for sid, row in zip(stocks, base):
                hist[int(sid)].append(row)
            seq = torch.tensor(np.stack([np.stack(hist[int(s)]) for s in stocks]))
        t["3_seq_build"] += time.perf_counter() - a

        a = time.perf_counter()
        if sgrus:
            p_gru = np.mean([s.step(x_t, sidx).numpy() for s in sgrus], axis=0)
        else:
            with torch.no_grad():
                p_gru = np.mean([g(seq, sidx)[:, -1].numpy() for g in grus], axis=0)
        t["4_gru_fwd"] += time.perf_counter() - a

        a = time.perf_counter()
        if stfms:
            p_tfm = np.mean([s.step(x_t, sidx).numpy() for s in stfms], axis=0)
        else:
            with torch.no_grad():
                p_tfm = np.mean([m(seq, sidx)[:, -1].numpy() for m in tfms], axis=0)
        t["5_tfm_fwd"] += time.perf_counter() - a

        a = time.perf_counter()
        blend = WL * p_lgb + WG * p_gru + WT * p_tfm
        blend = blend - blend.mean()
        sample["target"] = sample.row_id.map(pd.Series(blend, index=cur.row_id.values)).to_numpy()
        preds.append(sample)
        t["6_blend_assign"] += time.perf_counter() - a

        step_in_day += 1
        per_bucket.append((step_in_day, time.perf_counter() - b0))

    total = time.perf_counter() - t_start
    sub = pd.concat(preds).set_index("row_id")["target"]

    if profile:
        label = ("STATEFUL GRU + KV-CACHE TRANSFORMER" if stateful_gru and stateful_tfm
                 else "STATEFUL GRU" if stateful_gru
                 else "BASELINE (full re-forward)")
        print(f"\n===== {label} — total {total:.1f}s for {len(per_bucket)} buckets "
              f"({total/len(per_bucket)*1000:.0f} ms/bucket) =====")
        for k in sorted(t):
            print(f"  {k:16s} {t[k]:7.2f}s  ({t[k]/total*100:5.1f}%)")
        # cost growth within a day
        pb = pd.DataFrame(per_bucket, columns=["step_in_day", "sec"])
        early = pb[pb.step_in_day <= 5].sec.mean() * 1000
        late = pb[pb.step_in_day >= 50].sec.mean() * 1000
        print(f"  per-bucket cost: first 5 steps {early:.0f} ms  ->  last steps {late:.0f} ms "
              f"({late/early:.1f}x growth within a day)")
    return sub, total


if __name__ == "__main__":
    sub_base, t_base = run()
    sub_sgru, t_sgru = run(stateful_gru=True)
    sub_full, t_full = run(stateful_gru=True, stateful_tfm=True)
    print(f"\n===== SUMMARY (vs baseline) =====")
    print(f"stateful GRU only        : {t_base/t_sgru:.2f}x   max|diff| = {(sub_base - sub_sgru).abs().max():.3e}")
    print(f"stateful GRU + KV-cache  : {t_base/t_full:.2f}x   max|diff| = {(sub_base - sub_full).abs().max():.3e}")
    print("(max|diff| must be ~1e-6 or smaller: float32 rounding noise only)")

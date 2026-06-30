#!/usr/bin/env python3
"""
expN_ci_block_bootstrap.py
诗人级 block bootstrap：修正 §5.2 ci 内聚性 (Cohen's d=1.90) 的伪重复问题

问题（审稿意见 / 用户既定降级 b）：
  原 §5.2 的 d=1.90 与 Mann-Whitney p 值建立在 235,641 个 ci-ci「诗人对」上，
  但这些对仅来自 687 个 ci 诗人（每位诗人出现在 ~686 个对中）。pair-level 检验
  把高度非独立的诗人对当作独立样本，严重夸大显著性（伪重复 / pseudoreplication）。

修正：以「诗人」为重抽样单元的 block bootstrap。
  - 预计算两个互文距离矩阵（与 _archive/18 完全同构的 GPU 双侧归一化互文度量）：
      D_ci    : 全部 687 个 ci 诗人 → 687×687（即原文的 235,641 对，完整复现）
      D_nonci : 从 3945 个 non-ci 诗人中随机抽 S 个 → S×S
  - 每个 bootstrap 复本：对 ci 诗人 / non-ci 诗人分别「有放回」重抽样，
    在重抽样诗人集合内重算 ci-ci / nonci-nonci 平均互文距离与 Cohen's d。
  - 报告 d 与两组均值的诗人级 95% 百分位 CI，取代无效的 pair-level p 值。

度量定义与原文一致（保证点估计可比）：
  cohens_d = (nonci_mean - ci_mean) / std(nonci_distances)
  另附 pooled-SD 版本供参考。
"""

import os, json, random, time, itertools
import numpy as np
import torch

BASE      = "/home/chenhao/poetryCrystallization"
POET_JSON = f"{BASE}/data/processed/poet_poems.json"
GENRE     = f"{BASE}/data/processed/poet_genre_by_source.json"
EMB_NPZ   = f"{BASE}/data/processed/sentence_embeddings_by_poet.npz"
OUT_JSON  = f"{BASE}/data/processed/expN_ci_block_bootstrap.json"
CACHE_CI  = f"{BASE}/data/processed/expN_D_ci.npy"
CACHE_NC  = f"{BASE}/data/processed/expN_D_nonci.npy"
CACHE_NC_NAMES = f"{BASE}/data/processed/expN_nonci_names.json"

TAU       = 0.85
SAMPLE_N  = 500
BATCH     = 128
SEED      = 42
CI_THRESH = 5
S_NONCI   = 1000      # non-ci 诗人子样本规模（用于诗人级 block bootstrap）
N_BOOT    = 2000      # bootstrap 复本数

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device {device}] SEED={SEED}")

# ── 1. 分组（与 _archive/18 同口径：ci>=5）─────────────────────────
poets  = json.load(open(POET_JSON, encoding="utf-8"))
genres = json.load(open(GENRE, encoding="utf-8"))
npz    = np.load(EMB_NPZ, allow_pickle=True)   # 惰性加载，仅按需取 key
present = set(npz.files)

ci_n   = {p["name"]: genres.get(p["name"], {}).get("ci", 0) for p in poets}
dyn_of = {p["name"]: p.get("dynasty", "其他") for p in poets}

ci_poets = [p["name"] for p in poets if ci_n[p["name"]] >= CI_THRESH and p["name"] in present]
nonci    = [p["name"] for p in poets if ci_n[p["name"]] <  CI_THRESH and p["name"] in present]
print(f"ci>={CI_THRESH}: {len(ci_poets)}   non-ci: {len(nonci)}")

# non-ci 随机子样本（诗人级重抽样的总体），固定 seed 可复现
rng = random.Random(SEED)
nonci_sub = sorted(rng.sample(nonci, min(S_NONCI, len(nonci))))
print(f"non-ci 子样本: {len(nonci_sub)}（用于 block bootstrap）")

# ── 2. GPU 互文距离（双侧归一化，TAU=0.85）—— 与 _archive/18 同构 ──
TAU_T = torch.tensor(TAU, dtype=torch.float32, device=device)

def load_emb(names):
    return {nm: npz[nm].astype(np.float32) for nm in names}

def intertextual_matrix(names, emb, tag=""):
    """全对 (i<j) 互文距离 → 对称矩阵 (n×n)，对角线=0。"""
    n = len(names)
    idx_pairs = [(i, j) for i, j in itertools.combinations(range(n), 2)]
    D = np.zeros((n, n), np.float32)
    t0 = time.time()
    for s in range(0, len(idx_pairs), BATCH):
        chunk = idx_pairs[s:s+BATCH]
        A=[]; B=[]; Am=[]; Bm=[]
        for i, j in chunk:
            ea = emb[names[i]]; eb = emb[names[j]]
            na = min(len(ea), SAMPLE_N); nb = min(len(eb), SAMPLE_N)
            ia = random.sample(range(len(ea)), na); ib = random.sample(range(len(eb)), nb)
            Ap = np.zeros((SAMPLE_N, 512), np.float32); Bp = np.zeros((SAMPLE_N, 512), np.float32)
            Ap[:na] = ea[ia]; Bp[:nb] = eb[ib]
            A.append(Ap); B.append(Bp); Am.append(na); Bm.append(nb)
        A  = torch.tensor(np.stack(A), device=device)
        Bt = torch.tensor(np.stack(B), device=device)
        Am = torch.tensor(Am, device=device); Bm = torch.tensor(Bm, device=device)
        An = A  / (A.norm(dim=-1, keepdim=True)  + 1e-9)
        Bn = Bt / (Bt.norm(dim=-1, keepdim=True) + 1e-9)
        S_ = torch.bmm(An, Bn.transpose(-2, -1))
        above = (S_ > TAU_T).float()
        ii = torch.arange(SAMPLE_N, device=device).unsqueeze(0)
        av = (ii < Am.unsqueeze(1)).float(); bv = (ii < Bm.unsqueeze(1)).float()
        p_ab = (above.any(1) * bv).sum(1) / Bm.float()
        p_ba = (above.any(2) * av).sum(1) / Am.float()
        d = (1.0 - torch.sqrt(p_ab * p_ba + 1e-9)).cpu().numpy()
        for k, (i, j) in enumerate(chunk):
            D[i, j] = D[j, i] = d[k]
        done = s + len(chunk)
        if done % 20000 < BATCH or done == len(idx_pairs):
            print(f"    [{tag}] {done}/{len(idx_pairs)}  {done/(time.time()-t0):.0f} pairs/s")
    return D

# 复用缓存
if os.path.exists(CACHE_CI) and os.path.exists(CACHE_NC) and os.path.exists(CACHE_NC_NAMES):
    print("\n[cache] 载入已算距离矩阵")
    D_ci = np.load(CACHE_CI)
    D_nc = np.load(CACHE_NC)
    nonci_sub = json.load(open(CACHE_NC_NAMES, encoding="utf-8"))
    assert D_ci.shape[0] == len(ci_poets), "ci 缓存与当前诗人集不一致"
    assert D_nc.shape[0] == len(nonci_sub), "nonci 缓存与当前子样本不一致"
else:
    print("\n[2] GPU 计算 ci-ci 距离矩阵（687×687, 全部对）...")
    emb_ci = load_emb(ci_poets)
    D_ci = intertextual_matrix(ci_poets, emb_ci, "ci")
    del emb_ci
    np.save(CACHE_CI, D_ci)

    print(f"\n[3] GPU 计算 nonci 子样本距离矩阵（{len(nonci_sub)}×{len(nonci_sub)}）...")
    emb_nc = load_emb(nonci_sub)
    D_nc = intertextual_matrix(nonci_sub, emb_nc, "nonci")
    del emb_nc
    np.save(CACHE_NC, D_nc)
    json.dump(nonci_sub, open(CACHE_NC_NAMES, "w", encoding="utf-8"), ensure_ascii=False)

# ── 3. 上三角向量（点估计）──────────────────────────────────────
def upper_vals(D):
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]

ci_vals = upper_vals(D_ci)
nc_vals = upper_vals(D_nc)
ci_mean = float(ci_vals.mean()); nc_mean = float(nc_vals.mean())
d_orig  = (nc_mean - ci_mean) / nc_vals.std()
pooled  = np.sqrt((ci_vals.var(ddof=1) + nc_vals.var(ddof=1)) / 2)
d_pool  = (nc_mean - ci_mean) / pooled
print(f"\n点估计: ci_mean={ci_mean:.4f} (n={len(ci_vals):,})  "
      f"nonci_mean={nc_mean:.4f} (n={len(nc_vals):,})")
print(f"        Cohen's d (std-nonci, 原文定义)={d_orig:.3f}   pooled-SD={d_pool:.3f}")

# ── 4. 诗人级 block bootstrap ──────────────────────────────────
def block_means(D, n):
    """重抽样 n 个诗人(有放回)，在重抽样集合内取所有相异诗人对的均值/方差。
       同一诗人被抽多次时其自配对(对角=0)排除。"""
    idx = np.random.randint(0, n, size=n)
    sub = D[np.ix_(idx, idx)]
    iu = np.triu_indices(n, k=1)
    same = (idx[iu[0]] == idx[iu[1]])         # 同一原诗人的重复 → 排除自配对
    vals = sub[iu][~same]
    return vals.mean(), vals.var(ddof=1), len(vals)

n_ci = D_ci.shape[0]; n_nc = D_nc.shape[0]
np.random.seed(SEED)
boot_d = np.empty(N_BOOT); boot_dp = np.empty(N_BOOT)
boot_ci = np.empty(N_BOOT); boot_nc = np.empty(N_BOOT)
t0 = time.time()
for b in range(N_BOOT):
    cm, cv, _ = block_means(D_ci, n_ci)
    nm, nv, _ = block_means(D_nc, n_nc)
    boot_ci[b] = cm; boot_nc[b] = nm
    boot_d[b]  = (nm - cm) / np.sqrt(nv)
    boot_dp[b] = (nm - cm) / np.sqrt((cv + nv) / 2)
print(f"  {N_BOOT} bootstrap 复本  {time.time()-t0:.1f}s")

def ci95(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

d_lo, d_hi   = ci95(boot_d)
dp_lo, dp_hi = ci95(boot_dp)
cm_lo, cm_hi = ci95(boot_ci)
nm_lo, nm_hi = ci95(boot_nc)
diff = boot_nc - boot_ci
diff_lo, diff_hi = ci95(diff)

print("\n" + "="*70)
print(f"诗人级 block bootstrap 结果 (95pct 百分位 CI, B={N_BOOT})")
print("="*70)
print(f"  ci_mean      : {ci_mean:.4f}  CI[{cm_lo:.4f}, {cm_hi:.4f}]")
print(f"  nonci_mean   : {nc_mean:.4f}  CI[{nm_lo:.4f}, {nm_hi:.4f}]")
print(f"  均值差 (nonci-ci): {nc_mean-ci_mean:.4f}  CI[{diff_lo:.4f}, {diff_hi:.4f}]")
print(f"  Cohen's d (std-nonci): {d_orig:.3f}  CI[{d_lo:.3f}, {d_hi:.3f}]")
print(f"  Cohen's d (pooled-SD): {d_pool:.3f}  CI[{dp_lo:.3f}, {dp_hi:.3f}]")
print(f"  CI 下界是否 > 0 (效应方向稳健): {'是' if diff_lo > 0 else '否'}")

# ── 5. 朝代内 block bootstrap（取代 §5.2 朝代内 Mann-Whitney p 值）──
ci_dyn = np.array([dyn_of[n] for n in ci_poets])
nc_dyn = np.array([dyn_of[n] for n in nonci_sub])

def within_block(D, dyn, mask, n_boot=N_BOOT):
    """限定某朝诗人，block bootstrap 组内诗人对均值分布。"""
    idx_all = np.where(mask)[0]
    if len(idx_all) < 3:
        return None
    means = np.empty(n_boot)
    for b in range(n_boot):
        ridx = np.random.choice(idx_all, size=len(idx_all), replace=True)
        sub = D[np.ix_(ridx, ridx)]
        iu = np.triu_indices(len(ridx), k=1)
        same = (ridx[iu[0]] == ridx[iu[1]])
        v = sub[iu][~same]
        means[b] = v.mean() if len(v) else np.nan
    return means

within_boot = {}
np.random.seed(SEED)
print("\n── 朝代内 block bootstrap (ci - nonci 差值 95% CI) ──")
for dy in ["唐", "宋", "元", "明", "清", "近代"]:
    cm_ = within_block(D_ci, ci_dyn, ci_dyn == dy)
    nm_ = within_block(D_nc, nc_dyn, nc_dyn == dy)
    n_ci_d = int((ci_dyn == dy).sum()); n_nc_d = int((nc_dyn == dy).sum())
    if cm_ is None or nm_ is None:
        print(f"  {dy}: ci_poets={n_ci_d} nonci_poets={n_nc_d} (样本不足，跳过)")
        continue
    ci_pt = float(np.nanmean(D_ci[np.ix_(ci_dyn==dy, ci_dyn==dy)][np.triu_indices((ci_dyn==dy).sum(),1)]))
    nc_pt = float(np.nanmean(D_nc[np.ix_(nc_dyn==dy, nc_dyn==dy)][np.triu_indices((nc_dyn==dy).sum(),1)]))
    d_boot = nm_ - cm_
    lo, hi = ci95(d_boot[~np.isnan(d_boot)])
    robust = hi < 0  # nonci 远 → ci 更近，差(ci-nonci)<0；这里 d_boot=nonci-ci 应 >0
    within_boot[dy] = {
        "n_ci_poets": n_ci_d, "n_nonci_poets": n_nc_d,
        "ci_mean": ci_pt, "nonci_mean": nc_pt,
        "mean_diff_nonci_minus_ci": nc_pt - ci_pt,
        "diff_ci95": [lo, hi],
        "robust_ci_closer": bool(lo > 0)}
    print(f"  {dy}: ci_poets={n_ci_d} nonci_poets={n_nc_d}  "
          f"ci={ci_pt:.3f} nonci={nc_pt:.3f}  diff(nonci-ci)={nc_pt-ci_pt:+.3f} "
          f"CI[{lo:.3f},{hi:.3f}]  {'稳健' if lo>0 else '不稳健'}")

out = {
    "method": "poet-level block bootstrap (resample poets w/ replacement)",
    "metric": "bilateral normalized intertextual distance (TAU=0.85, SAMPLE_N=500)",
    "ci_threshold": CI_THRESH, "seed": SEED, "n_boot": N_BOOT,
    "n_ci_poets": n_ci, "n_nonci_poets_subsample": n_nc,
    "n_nonci_poets_total": len(nonci),
    "n_ci_pairs": int(len(ci_vals)), "n_nonci_pairs": int(len(nc_vals)),
    "point": {
        "ci_mean": ci_mean, "nonci_mean": nc_mean,
        "mean_diff": nc_mean - ci_mean,
        "cohens_d_std_nonci": float(d_orig), "cohens_d_pooled": float(d_pool)},
    "block_bootstrap_ci95": {
        "ci_mean": [cm_lo, cm_hi], "nonci_mean": [nm_lo, nm_hi],
        "mean_diff": [diff_lo, diff_hi],
        "cohens_d_std_nonci": [d_lo, d_hi],
        "cohens_d_pooled": [dp_lo, dp_hi]},
    "within_dynasty_block_bootstrap": within_boot,
    "effect_direction_robust": bool(diff_lo > 0),
    "note": ("原 §5.2 报告的 pair-level Cohen's d=1.90 与 Mann-Whitney p 值"
             "因伪重复（235,641 对仅来自 687 诗人）而无效；本表以诗人为重抽样"
             "单元的 block bootstrap CI 取代之。朝代内差值同样改用诗人级 CI。")}
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n✓ 已保存: {OUT_JSON}")

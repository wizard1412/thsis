import pandas as pd

df = pd.read_csv("all_approaches_summary.csv")
df.columns = [c.strip() for c in df.columns]
fs = {'ds32b_v1_amp': 'Consensus(amp)', 'ds32b_v1_llmfreq': 'LLM-freq'}

for app in ['A_concordant', 'B_mixed', 'C_all_halfup']:
    sub = df[df['approach'] == app]
    print(f"\n========== {app}  (n_samples={sub['n_samples'].iloc[0]}) ==========")
    for m in df['model'].unique():
        s = sub[sub['model'] == m]
        if s.empty:
            continue
        bk = s.loc[s['kappa'].idxmax()]
        bm = s.loc[s['mae'].idxmin()]
        be = s.loc[s['exact'].idxmax()]
        print(f"  {fs[m]:16s} best-kappa : n={int(bk.n):2d}  kappa={bk.kappa:.3f} mae={bk.mae:.3f} exact={bk.exact:.3f} adj={bk.within1:.3f}")
        print(f"  {'':16s} best-MAE   : n={int(bm.n):2d}  mae={bm.mae:.3f} kappa={bm.kappa:.3f} exact={bm.exact:.3f} adj={bm.within1:.3f}")
        print(f"  {'':16s} best-exact : n={int(be.n):2d}  exact={be.exact:.3f} kappa={be.kappa:.3f} mae={be.mae:.3f} adj={be.within1:.3f}")

print("\n=== rounding robustness (B vs C) ===")
b = df[df['approach'] == 'B_mixed'].set_index(['model', 'n'])['kappa']
c = df[df['approach'] == 'C_all_halfup'].set_index(['model', 'n'])['kappa']
print("  kappa(B)==kappa(C) for all configs?", bool((b.round(4) == c.round(4)).all()))
mae_b = df[df['approach'] == 'B_mixed']['mae'].mean()
mae_c = df[df['approach'] == 'C_all_halfup']['mae'].mean()
print(f"  mean MAE: B_mixed={mae_b:.3f}  C_halfup={mae_c:.3f}  diff={abs(mae_b-mae_c):.3f}")

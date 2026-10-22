import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = pd.read_csv("summary.csv")
t["m"] = t["model"].map({"twitter-roberta-base-hate": "Model 1", "unbiased-toxic-roberta": "Model 2"})
rows = list(t.iterrows())
fig, ax = plt.subplots(figsize=(4.6, 3.1))
labels = []
for i, (_, r) in enumerate(rows):
    y = len(rows) - 1 - i
    color = "#1f4e79" if r["m"] == "Model 1" else "#c55a11"
    marker = "o" if r["definition"] == "baseline" else "s"
    ax.errorbar(r["p_aae_higher"], y, xerr=[[r["p_aae_higher"] - r["p_low"]], [r["p_high"] - r["p_aae_higher"]]],
                fmt=marker, color=color, capsize=2, ms=4, lw=1)
    labels.append(f'{r["corpus"]}, {r["definition"]}, {r["m"][-1]}')
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels[::-1], fontsize=6.5)
ax.axvline(0.5, color="gray", ls="--", lw=0.8)
ax.set_xlim(0.45, 0.85); ax.set_xlabel("P(AAE score > mainstream score), tweets labelled 'neither'", fontsize=7)
ax.tick_params(axis="x", labelsize=7)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig("fig_pab.pdf")

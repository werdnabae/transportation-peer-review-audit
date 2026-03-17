"""
peer_review_analysis.py
Computes all statistics, tables, and logistic regression results
for: Peer Review Governance in Transportation Research
Data: transportation_journal_peer_review_dataset.xlsx
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
data_path = Path(__file__).parent / "transportation_journal_peer_review_dataset.xlsx"
df = pd.read_excel(data_path)
print(f"Dataset: {len(df)} journals\n")

# Binary outcome (exclude Unknown from inferential analysis)
coded = df[df["Peer_Review_Model"] != "Unknown"].copy()
coded["SB"] = (coded["Peer_Review_Model"] == "Single-blind").astype(int)
print(f"Coded (non-Unknown): {len(coded)} journals")

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 1 — Overall distribution
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 1: Overall PR Model Distribution")
print("=" * 60)
t1 = df["Peer_Review_Model"].value_counts().reset_index()
t1.columns = ["Peer_Review_Model", "n"]
t1["pct_all"] = (t1["n"] / len(df) * 100).round(1)
t1["pct_coded"] = (t1["n"] / len(coded) * 100).round(1)
print(t1.to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 2 — PR model by publisher
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 2: PR Model by Publisher")
print("=" * 60)
pub_pr = df.groupby(["Publisher", "Peer_Review_Model"]).size().unstack(fill_value=0)
# Add total and SB%
pub_pr["Total"] = pub_pr.sum(axis=1)
if "Single-blind" in pub_pr.columns:
    coded_total = pub_pr.get("Single-blind", 0) + pub_pr.get("Double-blind", 0)
    pub_pr["SB_pct"] = (
        pub_pr.get("Single-blind", 0) / coded_total.replace(0, np.nan) * 100
    ).round(1)
pub_pr = pub_pr.sort_values("Total", ascending=False)
print(pub_pr.to_string())

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 3 — PR model by impact tier
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 3: PR Model by Impact Tier")
print("=" * 60)
tier_order = ["High", "Medium", "Low"]
tier_pr = df.groupby(["Impact_Tier", "Peer_Review_Model"]).size().unstack(fill_value=0)
tier_pr = tier_pr.reindex(tier_order)
tier_pr["Total"] = tier_pr.sum(axis=1)
coded_total = tier_pr.get("Single-blind", 0) + tier_pr.get("Double-blind", 0)
tier_pr["SB_pct"] = (
    tier_pr.get("Single-blind", 0) / coded_total.replace(0, np.nan) * 100
).round(1)
print(tier_pr.to_string())

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 4 — PR model by OA status
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 4: PR Model by OA Status")
print("=" * 60)
oa_pr = df.groupby(["OA_Status", "Peer_Review_Model"]).size().unstack(fill_value=0)
oa_pr["Total"] = oa_pr.sum(axis=1)
coded_total = oa_pr.get("Single-blind", 0) + oa_pr.get("Double-blind", 0)
oa_pr["SB_pct"] = (
    oa_pr.get("Single-blind", 0) / coded_total.replace(0, np.nan) * 100
).round(1)
print(oa_pr.to_string())

# ──────────────────────────────────────────────────────────────────────────────
# TABLE 5 — PR model by founding decade
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLE 5: PR Model by Founding Decade")
print("=" * 60)
df["Decade"] = (df["Year_Founded"].astype(int) // 10) * 10
coded["Decade"] = (coded["Year_Founded"].astype(int) // 10) * 10
dec_pr = df.groupby(["Decade", "Peer_Review_Model"]).size().unstack(fill_value=0)
dec_pr["Total"] = dec_pr.sum(axis=1)
coded_total = dec_pr.get("Single-blind", 0) + dec_pr.get("Double-blind", 0)
dec_pr["SB_pct"] = (
    dec_pr.get("Single-blind", 0) / coded_total.replace(0, np.nan) * 100
).round(1)
print(dec_pr.to_string())

# ──────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION — predictors of SB
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    reg_df = coded.copy()

    # Publisher group: Elsevier, Taylor & Francis, MDPI, IEEE, ASCE, Other
    def pub_group(p):
        if p == "Elsevier":
            return "Elsevier"
        if p == "Taylor & Francis":
            return "Taylor_Francis"
        if p == "MDPI":
            return "MDPI"
        if p == "IEEE":
            return "IEEE"
        if p == "ASCE":
            return "ASCE"
        if p == "SAE International":
            return "SAE"
        return "Other"

    reg_df["PubGroup"] = reg_df["Publisher"].apply(pub_group)

    # Impact tier numeric
    tier_map = {"High": 3, "Medium": 2, "Low": 1}
    reg_df["TierNum"] = reg_df["Impact_Tier"].map(tier_map)

    # OA binary
    reg_df["IsFullOA"] = (reg_df["OA_Status"].isin(["FullOA", "Diamond"])).astype(int)

    # Founding era
    reg_df["Post2000"] = (reg_df["Year_Founded"] >= 2000).astype(int)

    # Dummies for publisher group (reference = SAE / double-blind baseline)
    pub_dummies = pd.get_dummies(reg_df["PubGroup"], drop_first=False)

    feature_cols = ["TierNum", "IsFullOA", "Post2000"]
    X = pd.concat([reg_df[feature_cols], pub_dummies], axis=1)
    X = X.drop(columns=["SAE"], errors="ignore")  # reference category
    X = X.fillna(0)
    y = reg_df["SB"]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_sc, y)

    coef_df = pd.DataFrame({"Feature": X.columns, "Coef": lr.coef_[0]}).sort_values(
        "Coef", ascending=False
    )
    print(coef_df.to_string(index=False))

    # Simple chi-square tests
    print("\n--- Chi-square tests ---")

    # SB by publisher group
    pub_ct = pd.crosstab(coded["Publisher"].apply(pub_group), coded["SB"])
    chi2, p, dof, _ = stats.chi2_contingency(pub_ct)
    print(f"Publisher group vs SB: chi2={chi2:.2f}, df={dof}, p={p:.4f}")

    # SB by impact tier
    tier_ct = pd.crosstab(coded["Impact_Tier"], coded["SB"])
    chi2_t, p_t, dof_t, _ = stats.chi2_contingency(tier_ct)
    print(f"Impact tier vs SB:     chi2={chi2_t:.2f}, df={dof_t}, p={p_t:.4f}")

    # SB by OA status
    coded["OA_binary"] = coded["OA_Status"].apply(
        lambda x: "FullOA" if x in ["FullOA", "Diamond"] else "Subscription"
    )
    oa_ct = pd.crosstab(coded["OA_binary"], coded["SB"])
    chi2_o, p_o, dof_o, _ = stats.chi2_contingency(oa_ct)
    print(f"OA status vs SB:       chi2={chi2_o:.2f}, df={dof_o}, p={p_o:.4f}")

except ImportError:
    print("sklearn not available — install with: pip install scikit-learn")

# ──────────────────────────────────────────────────────────────────────────────
# KEY NUMBERS FOR PAPER
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY NUMBERS FOR PAPER")
print("=" * 60)
n = len(df)
n_coded = len(coded)
n_sb = (df["Peer_Review_Model"] == "Single-blind").sum()
n_db = (df["Peer_Review_Model"] == "Double-blind").sum()
n_unk = (df["Peer_Review_Model"] == "Unknown").sum()

print(f"N total:              {n}")
print(f"N coded:              {n_coded}")
print(
    f"N single-blind:       {n_sb} ({n_sb / n * 100:.1f}% of all; {n_sb / n_coded * 100:.1f}% of coded)"
)
print(
    f"N double-blind:       {n_db} ({n_db / n * 100:.1f}% of all; {n_db / n_coded * 100:.1f}% of coded)"
)
print(f"N unknown:            {n_unk} ({n_unk / n * 100:.1f}%)")

elv = df[df["Publisher"] == "Elsevier"]
tf = df[df["Publisher"] == "Taylor & Francis"]
mdpi = df[df["Publisher"] == "MDPI"]
ieee = df[df["Publisher"] == "IEEE"]
asce = df[df["Publisher"] == "ASCE"]
sae = df[df["Publisher"] == "SAE International"]
informs = df[df["Publisher"] == "INFORMS"]
high_df = coded[coded["Impact_Tier"] == "High"]

for name, grp in [
    ("Elsevier", elv),
    ("T&F", tf),
    ("MDPI", mdpi),
    ("IEEE", ieee),
    ("ASCE", asce),
    ("SAE", sae),
    ("INFORMS", informs),
]:
    sb_n = (grp["Peer_Review_Model"] == "Single-blind").sum()
    db_n = (grp["Peer_Review_Model"] == "Double-blind").sum()
    tot = len(grp)
    coded_n = sb_n + db_n
    pct = sb_n / coded_n * 100 if coded_n > 0 else 0
    print(f"{name:15} n={tot:2}  SB={sb_n} ({pct:.0f}%)  DB={db_n}")

print(f"\nHigh-impact (n={len(high_df)} coded):")
print(f"  SB: {high_df['SB'].sum()} ({high_df['SB'].mean() * 100:.1f}%)")
print(f"  DB: {(high_df['SB'] == 0).sum()} ({(high_df['SB'] == 0).mean() * 100:.1f}%)")

print("\nScript complete.")

import pandas as pd
import re

df = pd.read_parquet("GPT_annotated_data.parquet")

print("Colonnes trouvées :", list(df.columns))

# =========================
# Trouver la colonne score
# =========================
score_col = None

for col in df.columns:
    sample = df[col].dropna().astype(str).head(20).tolist()
    if any(re.search(r'[A-Z]\d', s) for s in sample):
        score_col = col
        break

if score_col is None:
    raise ValueError("❌ Aucune colonne de scores (S0/H2/...) trouvée")

print("✅ Colonne de scores détectée :", score_col)

# =========================
# Création du label
# =========================
def make_label(x):
    scores = re.findall(r'[A-Z](\d+)', str(x))
    return 1 if any(int(s) > 0 for s in scores) else 0

df_final = pd.DataFrame({
    "text": df["content"],
    "label": df[score_col].apply(make_label)
})

# Nettoyage
df_final = df_final.dropna(subset=["text"])

# Stats
print("\n=== Stats du dataset ===")
print(df_final["label"].value_counts())
print("Taux toxique :", df_final["label"].mean() * 100, "%")

# Export
df_final.to_csv("dataset_toxicity.csv", index=False, encoding="utf-8")

print("\n✅ Dataset créé : dataset_toxicity.csv")

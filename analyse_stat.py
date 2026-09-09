import pandas as pd

# Lecture du fichier
df = pd.read_csv("regions_DHMS.tsv", sep="\t")

# Longueur de chaque région
df["longueur"] = df["end"] - df["start"]

# Nombre de bins de 1 kb par région
df["nb_bins"] = df["longueur"] // 1000

print("\nNOMBRE DE RÉGIONS")

print(df["state"].value_counts())

print("\nLONGUEUR DES RÉGIONS")

for etat in ["ESC", "NPC"]:

    sous_df = df[df["state"] == etat]

    print(f"\n{etat}")
    print("Nombre :", len(sous_df))
    print("Longueur moyenne :", sous_df["longueur"].mean(), "bp")
    print("Longueur médiane :", sous_df["longueur"].median(), "bp")
    print("Longueur minimale :", sous_df["longueur"].min(), "bp")
    print("Longueur maximale :", sous_df["longueur"].max(), "bp")
    print("Nombre moyen de bins :", sous_df["nb_bins"].mean())
    print("Nombre médian de bins :", sous_df["nb_bins"].median())

print("\nRÉGIONS LES PLUS LONGUES")

print(
    df.sort_values("longueur", ascending=False)
      .head(10)
      [["chromosome", "start", "end", "longueur", "nb_bins", "state"]]
)
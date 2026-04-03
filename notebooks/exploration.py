import pandas as pd

df = pd.read_csv("data/patients_dakar.csv")

print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

print("Nombre de patients : " + str(len(df)))
print("Nombre de colonnes : " + str(df.shape[1]))

print("\n--- 5 premiers patients ---")
print(df.head())

print("\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(diag + " : " + str(count) + " patients (" + str(round(pct,1)) + "%)")

print("\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(diag + " : " + str(round(temp,1)) + " C")

print("=" * 50)
print("Exploration terminee !")
print("=" * 50)
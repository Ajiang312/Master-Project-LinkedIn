import pandas as pd
import glob
import os

# Lister tous les fichiers CSV ciblés
csv_files = glob.glob("linkedin-ingenieur-database*.csv")

fusion = pd.DataFrame()
for file in csv_files:
    try:
        df = pd.read_csv(file, encoding="utf-8-sig")
        if "job_id" in df.columns:
            fusion = pd.concat([fusion, df], ignore_index=True)
            print(f"✅ Chargé : {file} ({len(df)} lignes)")
        else:
            print(f"⚠️ Ignoré : {file} (pas de colonne 'job_id')")
    except Exception as e:
        print(f"❌ Erreur fichier {file} : {e}")

# Suppression des doublons selon job_id uniquement
fusion_cleaned = fusion.drop_duplicates(subset="job_id")

# Export final
output_file = "linkedin-ingenieur-database-merged.csv"
fusion_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig", quoting=1)

print(f"\n📝 Fichier fusionné créé : {output_file} ({len(fusion_cleaned)} lignes uniques)")

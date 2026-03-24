

from libraries import *

feats = ["HR", "O2Sat", "Temp", "SBP", "DBP", "Resp","MAP" ,"SepsisLabel"]
imputed_data_dir=kf_data_dir
psv_files = sorted(glob.glob(os.path.join(imputed_data_dir, "*.psv")))
print("Total files:", len(psv_files))

updated = 0
for fn in psv_files:
    df = pd.read_csv(fn, sep="|", na_values=["", "NaN", "nan"], keep_default_na=True)

    for c in feats:
        if c not in df.columns:
            df[c] = np.nan

    df_out = df.loc[:, feats].copy()

    tmp_path = fn + ".tmp"
    df_out.to_csv(tmp_path, sep="|", index=False)
    os.replace(tmp_path, fn)

    updated += 1

print("Updated files:", updated)

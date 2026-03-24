
from libraries import *

extra_dir="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/extra"
main_dir=extra_dir
time = 48
label = "SepsisLabel"
psv_files = sorted(glob.glob(os.path.join(main_dir, "*.psv")))
print("Total files:", len(psv_files))
if not psv_files:
    raise FileNotFoundError(f"No .psv files found in: {main_dir}")

updated = 0
skipped = 0

for fn in psv_files:
    df = pd.read_csv(fn, sep="|", na_values=["", "NaN", "nan"], keep_default_na=True)

    if df is None or df.shape[0] == 0 or label not in df.columns:
        skipped += 1
        continue

    n = len(df)

    if n > time:
        df_out = df.tail(time).reset_index(drop=True)

    elif n < time:
        pad_n = time - n

        last_label = df[label].iloc[-1]   

        pad = pd.DataFrame(np.nan, index=range(pad_n), columns=df.columns)
        pad[label] = last_label
        
        
        df_out = pd.concat([pad, df.reset_index(drop=True)], ignore_index=True) 
        # df_out = pd.concat([df.reset_index(drop=True), pad], ignore_index=True)
    else:
        df_out = df.reset_index(drop=True)

    df_out.to_csv(fn, sep="|", index=False)
    updated += 1

print("\nDone.")
print("Updated files:", updated)
print("Skipped:", skipped)
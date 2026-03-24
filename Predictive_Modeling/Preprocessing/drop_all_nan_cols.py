from libraries import *

extra_dir="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/extra"
main_dir=extra_dir
psv_files = sorted(glob.glob(os.path.join(main_dir, "*.psv")))
print("Total files:", len(psv_files))
if not psv_files:
    raise FileNotFoundError(f"No files found in: {main_dir}")

deleted = 0
skipped = 0

for fn in psv_files:
    df = pd.read_csv(fn, sep="|", na_values=["", "NaN", "nan"], keep_default_na=True)

    if any(c not in df.columns for c in feat):
        skipped += 1
        continue
    
    nan_cols = df[feat].isna().all(axis=0).sum()
    if nan_cols >= 1:
        os.remove(fn)
        deleted += 1

remaining = sorted(glob.glob(os.path.join(main_dir, "*.psv")))

print("Deleted: ", deleted)
print("Skipped: ", skipped)
print("Total patients left: ", len(remaining))

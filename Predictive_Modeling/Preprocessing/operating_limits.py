from libraries import *

extra_dir="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/extra"
main_dir=extra_dir
psv_files = sorted(glob.glob(os.path.join(main_dir, "*.psv")))

limits = {
    "HR":    (30, 220),
    "MAP":    (40, 160),
    "O2Sat": (70, 100),
    "Temp":  (32, 42),
    "SBP":   (40, 260),
    "DBP":   (20, 150),
    "Resp":  (4,50),
}




print("Total files:", len(psv_files))

n_updated = 0
n_bad = 0
n_skipped_missing_cols = 0

total_replaced = {k: 0 for k in limits.keys()}

for fn in psv_files:
    try:
        df = pd.read_csv(fn, sep="|", na_values=["", "NaN", "nan"], keep_default_na=True)
    except Exception:
        n_bad += 1
        continue

    changed_any = False
    missing_any = False

    for col, (lo, hi) in limits.items():
        if col not in df.columns:
            missing_any = True
            continue

        x = pd.to_numeric(df[col], errors="coerce")  # numeric for comparisons

        # ONLY replace values under lo or over hi (ignore NaNs automatically)
        out = False
        out = (x < lo) | (x > hi)

        if out.any():
            df.loc[out, col] = np.nan
            c = int(out.sum())
            total_replaced[col] += c
            changed_any = True

    if missing_any:
        n_skipped_missing_cols += 1

    if changed_any:
        df.to_csv(fn, sep="|", index=False)
        n_updated += 1

print("\nDone.")
print("Files updated:", n_updated)
print("Files failed to read:", n_bad)
print("Files missing at least one limit column:", n_skipped_missing_cols)

print("\nTotal values replaced with NaN (across all files):")
for k, v in total_replaced.items():
    print(f"  {k}: {v}")

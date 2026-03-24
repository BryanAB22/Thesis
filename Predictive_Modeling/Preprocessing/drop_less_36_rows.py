from libraries import *

extra_dir="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/extra"
main_dir=extra_dir
psv_files = sorted(glob.glob(os.path.join(main_dir, "*.psv")))
print("Total files:", len(psv_files))
if not psv_files:
    raise FileNotFoundError(f"No .psv files found in: {main_dir}")
min_lenght = 36
deleted = 0
kept_files = []

for fn in psv_files:
    df = pd.read_csv(fn, sep="|", na_values=["", "NaN", "nan"], keep_default_na=True)
    if len(df) < min_lenght:
        os.remove(fn)
        deleted += 1
    else:
        kept_files.append(fn)
remaining = sorted(glob.glob(os.path.join(main_dir, "*.psv")))

print("Deleted (<36 rows):", deleted)
print("Total patients left: ", len(remaining))

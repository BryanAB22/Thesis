from libraries import *
import os, glob
import numpy as np
import pandas as pd

out = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data"

sep = "|"
label_col = "SepsisLabel"

ENFORCE_48 = True
NROWS_PER_PATIENT = 48

out_dir = os.path.join(kf_data_dir, "CombinedCSVs")
os.makedirs(out_dir, exist_ok=True)

out_septic = os.path.join(out, "septic.csv")
out_non    = os.path.join(out, "nonseptic.csv")

for p in [out_septic, out_non]:
    if os.path.exists(p):
        os.remove(p)

psv_files = sorted(glob.glob(os.path.join(kf_data_dir, "*.psv")))
print("Total .psv files:", len(psv_files))
if not psv_files:
    raise FileNotFoundError(f"No .psv files found in: {kf_data_dir}")

# -------------------------
# Get reference columns by reading the first line (header) directly
# -------------------------
ref_cols = None
n_empty_or_bad_header = 0

for fn in psv_files:
    try:
        with open(fn, "r", encoding="utf-8", errors="replace") as f:
            # find first non-empty line
            while True:
                line = f.readline()
                if line == "":  # EOF
                    break
                line = line.strip("\n\r")
                if line.strip() != "":
                    break

        if not line or line.strip() == "":
            n_empty_or_bad_header += 1
            continue

        # strip UTF-8 BOM if present
        line = line.lstrip("\ufeff")

        # infer delimiter if needed (but you said it's '|')
        delim = sep if (sep in line) else ("|" if "|" in line else ("," if "," in line else ("\t" if "\t" in line else sep)))
        cols = [c.strip() for c in line.split(delim)]

        # basic sanity
        if len(cols) < 2:
            n_empty_or_bad_header += 1
            continue

        ref_cols = cols
        break

    except Exception:
        n_empty_or_bad_header += 1
        continue

if ref_cols is None:
    raise RuntimeError(
        "Could not read a header line from any .psv file. "
        "Possibles: files are empty, not text, or delimiter/header is missing."
    )

# ensure label exists in output columns
if label_col not in ref_cols:
    ref_cols.append(label_col)

print("Reference columns found:", len(ref_cols))
print("Files skipped while finding header:", n_empty_or_bad_header)

# -------------------------
# Combine into two CSVs (septic vs nonseptic)
# -------------------------
first_write_septic = True
first_write_non = True

n_written_septic = 0
n_written_non = 0
n_skipped_missing_label = 0
n_bad_reads = 0
n_fixed_rows = 0

for fn in psv_files:
    try:
        # engine="python" is more forgiving with weird formatting
        df = pd.read_csv(fn, sep=sep, na_values=["", "NaN", "nan"], keep_default_na=True, engine="python")
    except Exception:
        n_bad_reads += 1
        continue

    if label_col not in df.columns:
        n_skipped_missing_label += 1
        continue

    # align columns to ref
    for c in ref_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ref_cols]

    # enforce 48 rows per patient
    if ENFORCE_48:
        if len(df) > NROWS_PER_PATIENT:
            df = df.iloc[-NROWS_PER_PATIENT:].copy()
            n_fixed_rows += 1
        elif len(df) < NROWS_PER_PATIENT:
            pad = pd.DataFrame({c: [np.nan] * (NROWS_PER_PATIENT - len(df)) for c in ref_cols})
            df = pd.concat([df, pad], ignore_index=True)
            n_fixed_rows += 1

    # septic if ANY row has label==1
    s = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    is_septic = (s == 1).any()

    if is_septic:
        df.to_csv(out_septic, index=False, header=first_write_septic, mode="a")
        first_write_septic = False
        n_written_septic += 1
    else:
        df.to_csv(out_non, index=False, header=first_write_non, mode="a")
        first_write_non = False
        n_written_non += 1

print("\n=== DONE ===")
print("Septic combined CSV:    ", out_septic)
print("Non-septic combined CSV:", out_non)
print("\nCounts:")
print("  Septic patients appended:     ", n_written_septic)
print("  Non-septic patients appended: ", n_written_non)
print("  Skipped (missing SepsisLabel):", n_skipped_missing_label)
print("  Failed reads:", n_bad_reads)
print("  48-row fixes:", n_fixed_rows)

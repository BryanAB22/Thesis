from libraries import *


out_dir    = kf_data_dir
VS_COLS    = feat
SEPSIS_COL = "SepsisLabel"
imputed_data_dir= normalize_dir
os.makedirs(out_dir, exist_ok=True)

psv_files = sorted(glob.glob(os.path.join(imputed_data_dir, "*.psv")))

def first_onset_idx(df):
    s = df[SEPSIS_COL].astype(int).to_numpy()
    idx = np.where(s == 1)[0]
    return int(idx[0]) if idx.size else None


onset_vecs = []

for fn in psv_files:
    df = pd.read_csv(fn, sep="|")
    onset = first_onset_idx(df)
    if onset is None:
        continue
    onset_vecs.append(df.loc[onset, VS_COLS].to_numpy(dtype=float))

if not onset_vecs:
    raise RuntimeError("No septic patients found (no files with SepsisLabel==1).")

onset_mat = np.vstack(onset_vecs)
B = np.median(onset_mat, axis=0)  # sepsis position

print("Sepsis position B:")
print(dict(zip(VS_COLS, B)))

pd.DataFrame([B], columns=VS_COLS).to_csv(os.path.join(out_dir, "sepsis_position_B.csv"), index=False)

# Create KF features

written, skipped = 0, 0

for fn in psv_files:
    df = pd.read_csv(fn, sep="|")

    if any(c not in df.columns for c in VS_COLS) or SEPSIS_COL not in df.columns:
        skipped += 1
        continue

    X = df[VS_COLS].to_numpy(dtype=float)

    R = X - B[None, :]                           # relative position to sepsis position
    dist = np.linalg.norm(R, axis=1)             # ||R||
    e = np.zeros_like(R)
    nz = dist > 0
    e[nz] = R[nz] / dist[nz, None]               # unit direction vector

    V = np.zeros_like(R)
    V[1:] = R[1:] - R[:-1]                       # relative velocity (Δt = 1)

    A = np.zeros_like(R)
    A[1:] = V[1:] - V[:-1]                       # relative acceleration (Δt = 1)

    proj_v = np.sum(V * e, axis=1)               # projection of v onto e
    proj_a = np.sum(A * e, axis=1)               # projection of a onto e

    out = df.copy()
    for j, c in enumerate(VS_COLS):
        out[f"e.{c}"] = e[:, j]
    out["proj_v"] = proj_v
    out["proj_a"] = proj_a

    # out["dist_to_sepsis"] = dist

    out_path = os.path.join(out_dir, os.path.basename(fn))
    out.to_csv(out_path, sep="|", index=False)
    written += 1

print("\nDone.")
print("Wrote:", written, "files to", out_dir)
print("Skipped:", skipped)
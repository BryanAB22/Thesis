# impute_brits_dir.py : write the imputad .psv files

import os
import json
import glob
import numpy as np
import pandas as pd
import torch


from dataset import read_patient_window, normalize_array, denormalize_array
from brits_model import BRITSImputer


def main():
    model_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/BRITS"
    input_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_avg"
    output_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_BRITS"

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(model_dir, "config.json"), "r") as f:
        cfg = json.load(f)

    feat_cols = cfg["feat_cols"]
    T = cfg["T"]
    keep_last = cfg["keep_last"]
    normalize = cfg["normalize"]

    mean = np.load(os.path.join(model_dir, "train_mean.npy"))
    std = np.load(os.path.join(model_dir, "train_std.npy"))

    ckpt = torch.load(os.path.join(model_dir, "best.pt"), map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BRITSImputer(
        n_features=len(feat_cols),
        hidden_size=ckpt["hidden_size"],
        consistency_weight=ckpt["consistency_weight"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    files = sorted(glob.glob(os.path.join(input_dir, "*.psv")))
    if not files:
        raise FileNotFoundError(f"No .psv files found in: {input_dir}")

    written = 0

    with torch.no_grad():
        for fp in files:
            df_window, x = read_patient_window(
                fp,
                feat_cols=feat_cols,
                T=T,
                keep_last=keep_last
            )

            x_in = x.copy()
            if normalize:
                x_in = normalize_array(x_in, mean, std)

            x_tensor = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)  # [1, T, F]
            out = model(x_tensor)
            x_imp = out["imputed"].squeeze(0).cpu().numpy()  # [T, F]

            if normalize:
                x_imp = denormalize_array(x_imp, mean, std)

            df_out = df_window.copy()
            for j, c in enumerate(feat_cols):
                df_out[c] = x_imp[:, j]

            out_path = os.path.join(output_dir, os.path.basename(fp))
            df_out.to_csv(out_path, sep="|", index=False)
            written += 1

    print(f"Finished. Wrote {written} imputed files to:\n{output_dir}")


if __name__ == "__main__":
    main()
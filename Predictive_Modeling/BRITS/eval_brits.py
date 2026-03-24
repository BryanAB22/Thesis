# eval_brits.py : check MAE/RMSE (2)

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import list_psv_files, split_files, PatientWindowDataset
from brits_model import BRITSImputer, evaluate_on_holdout


def main():
    model_dir = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/BRITS"
    cfg_path = os.path.join(model_dir, "config.json")
    best_ckpt_path = os.path.join(model_dir, "best.pt")

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    feat_cols = cfg["feat_cols"]
    T = cfg["T"]
    keep_last = cfg["keep_last"]
    normalize = cfg["normalize"]
    dataset_dir = cfg["dataset_dir"]
    train_frac = cfg["train_frac"]
    seed = cfg["seed"]
    batch_size = cfg["batch_size"]
    holdout_frac = cfg["holdout_frac"]

    mean = np.load(os.path.join(model_dir, "train_mean.npy"))
    std = np.load(os.path.join(model_dir, "train_std.npy"))

    all_files = list_psv_files(dataset_dir)
    _, val_files = split_files(all_files, train_frac=train_frac, seed=seed)

    val_ds = PatientWindowDataset(
        val_files,
        feat_cols=feat_cols,
        T=T,
        keep_last=keep_last,
        normalize=normalize,
        mean=mean,
        std=std,
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    ckpt = torch.load(best_ckpt_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BRITSImputer(
        n_features=len(feat_cols),
        hidden_size=ckpt["hidden_size"],
        consistency_weight=ckpt["consistency_weight"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    maes = []
    rmses = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            res = evaluate_on_holdout(model, x, holdout_frac=holdout_frac)
            maes.append(res["mae"].item())
            rmses.append(res["rmse"].item())

    print("\n========== BRITS EVALUATION ==========")
    print(f"n_val_files: {len(val_ds)}")
    print(f"holdout_frac: {holdout_frac:.2f}")
    print(f"MAE:  {np.mean(maes):.6f}")
    print(f"RMSE: {np.mean(rmses):.6f}")


if __name__ == "__main__":
    main()
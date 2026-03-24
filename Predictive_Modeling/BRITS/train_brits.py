# train_brits.py : learn the imputation model (1)

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import BritsConfig
from dataset import (
    set_seed,
    list_psv_files,
    split_files,
    compute_train_stats,
    PatientWindowDataset,
)
from brits_model import BRITSImputer, training_step_loss, evaluate_on_holdout


def main():
    cfg = BritsConfig(
        dataset_dir="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_avg",
        out_dir="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/BRITS",
        feat_cols=["HR", "O2Sat", "Temp", "SBP", "DBP", "Resp","MAP"],
        T=48,
        keep_last=True,
        normalize=True,
        hidden_size=32,
        batch_size=32,
        epochs=50,
        lr=0.0025843981749201933,
        weight_decay=1e-5,
        holdout_frac=0.10,
        train_frac=0.80,
        seed=123,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    all_files = list_psv_files(cfg.dataset_dir)
    train_files, val_files = split_files(all_files, train_frac=cfg.train_frac, seed=cfg.seed)

    print(f"Total files: {len(all_files)}")
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")

    if cfg.normalize:
        mean, std = compute_train_stats(
            train_files,
            feat_cols=cfg.feat_cols,
            T=cfg.T,
            keep_last=cfg.keep_last
        )
    else:
        mean = np.zeros(len(cfg.feat_cols), dtype=np.float32)
        std = np.ones(len(cfg.feat_cols), dtype=np.float32)

    np.save(os.path.join(cfg.out_dir, "train_mean.npy"), mean)
    np.save(os.path.join(cfg.out_dir, "train_std.npy"), std)

    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    train_ds = PatientWindowDataset(
        train_files,
        feat_cols=cfg.feat_cols,
        T=cfg.T,
        keep_last=cfg.keep_last,
        normalize=cfg.normalize,
        mean=mean,
        std=std,
    )

    val_ds = PatientWindowDataset(
        val_files,
        feat_cols=cfg.feat_cols,
        T=cfg.T,
        keep_last=cfg.keep_last,
        normalize=cfg.normalize,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BRITSImputer(
        n_features=len(cfg.feat_cols),
        hidden_size=cfg.hidden_size,
        consistency_weight=cfg.consistency_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_mae = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        train_recon = []

        for batch in train_loader:
            x = batch["x"].to(device)  # [B, T, F], contains NaN

            optimizer.zero_grad()
            loss, out, target_mask, metrics = training_step_loss(
                model,
                x,
                holdout_frac=cfg.holdout_frac
            )
            loss.backward()
            optimizer.step()

            train_losses.append(metrics["total_loss"].item())
            train_recon.append(metrics["recon_mae"].item())

        model.eval()
        val_maes = []
        val_rmses = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                res = evaluate_on_holdout(model, x, holdout_frac=cfg.holdout_frac)
                val_maes.append(res["mae"].item())
                val_rmses.append(res["rmse"].item())

        train_loss_mean = float(np.mean(train_losses))
        train_recon_mean = float(np.mean(train_recon))
        val_mae_mean = float(np.mean(val_maes))
        val_rmse_mean = float(np.mean(val_rmses))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss_mean:.6f} | "
            f"train_recon_mae={train_recon_mean:.6f} | "
            f"val_mae={val_mae_mean:.6f} | "
            f"val_rmse={val_rmse_mean:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_mae": val_mae_mean,
            "val_rmse": val_rmse_mean,
            "feat_cols": cfg.feat_cols,
            "hidden_size": cfg.hidden_size,
            "consistency_weight": cfg.consistency_weight,
            "T": cfg.T,
        }

        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pt"))

        if val_mae_mean < best_val_mae:
            best_val_mae = val_mae_mean
            torch.save(ckpt, os.path.join(cfg.out_dir, "best.pt"))
            print(f"  Saved new best model with val_mae={best_val_mae:.6f}")


if __name__ == "__main__":
    main()
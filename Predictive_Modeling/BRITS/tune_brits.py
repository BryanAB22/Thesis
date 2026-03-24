import os
import json
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from dataset import (
    set_seed,
    list_psv_files,
    split_files,
    compute_train_stats,
    PatientWindowDataset,
)
from brits_model import BRITSImputer, training_step_loss, evaluate_on_holdout


DATASET_DIR = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_avg"
OUT_DIR = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/BRITS_TUNING"
FEAT_COLS = ["HR", "O2Sat", "Temp", "SBP", "DBP", "Resp","MAP"]

T = 48
KEEP_LAST = True
NORMALIZE = True
TRAIN_FRAC = 0.70
SEED = 123
NUM_WORKERS = 0

SEARCH_EPOCHS = 50
N_TRIALS = 5


os.makedirs(OUT_DIR, exist_ok=True)


def build_loaders(batch_size: int):
    all_files = list_psv_files(DATASET_DIR)
    train_files, val_files = split_files(all_files, train_frac=TRAIN_FRAC, seed=SEED)

    if NORMALIZE:
        mean, std = compute_train_stats(
            train_files,
            feat_cols=FEAT_COLS,
            T=T,
            keep_last=KEEP_LAST
        )
    else:
        mean = np.zeros(len(FEAT_COLS), dtype=np.float32)
        std = np.ones(len(FEAT_COLS), dtype=np.float32)

    train_ds = PatientWindowDataset(
        train_files,
        feat_cols=FEAT_COLS,
        T=T,
        keep_last=KEEP_LAST,
        normalize=NORMALIZE,
        mean=mean,
        std=std,
    )

    val_ds = PatientWindowDataset(
        val_files,
        feat_cols=FEAT_COLS,
        T=T,
        keep_last=KEEP_LAST,
        normalize=NORMALIZE,
        mean=mean,
        std=std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader


def objective(trial: optuna.Trial) -> float:
    print(f"\nStarting trial {trial.number}")

    set_seed(SEED)

    # hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 96, 128])
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    # weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    weight_decay= 1e-5
    consistency_weight= .1
    # consistency_weight = trial.suggest_float("consistency_weight", 1e-2, 5e-1, log=True)
    holdout_frac = .1

    train_loader, val_loader = build_loaders(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BRITSImputer(
        n_features=len(FEAT_COLS),
        hidden_size=hidden_size,
        consistency_weight=consistency_weight,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_val_mae = float("inf")

    for epoch in range(SEARCH_EPOCHS):
        model.train()

        for batch in train_loader:
            x = batch["x"].to(device)

            optimizer.zero_grad()
            loss, out, target_mask, metrics = training_step_loss(
                model,
                x,
                holdout_frac=holdout_frac
            )
            loss.backward()
            optimizer.step()

        model.eval()
        val_maes = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                res = evaluate_on_holdout(model, x, holdout_frac=holdout_frac)
                val_maes.append(res["mae"].item())

        mean_val_mae = float(np.mean(val_maes))
        best_val_mae = min(best_val_mae, mean_val_mae)

        trial.report(mean_val_mae, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_mae


def main():
    study = optuna.create_study(
        direction="minimize",
        study_name="brits_imputation_mae",
        storage=f"sqlite:///{os.path.join(OUT_DIR, 'brits_optuna.db')}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=5),
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest value (val_mae):", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)


if __name__ == "__main__":
    main()
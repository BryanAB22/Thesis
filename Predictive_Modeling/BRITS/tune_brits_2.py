# tune_brits_2.py

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


# =========================================================
# Paths / Features
# =========================================================
DATASET_DIR = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/Dataset_avg"
OUT_DIR = "/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/BRITS_TUNING_2STAGE"
FEAT_COLS = ["HR", "O2Sat", "Temp", "SBP", "DBP", "Resp", "MAP"]

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# Global settings
# =========================================================
T = 48
KEEP_LAST = True
NORMALIZE = True
TRAIN_FRAC = 0.80
SEED = 123
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Stage 1 (subset) settings
# =========================================================
SUBSET_FRAC_TRAIN = 0.20      # 20% of train files
SUBSET_FRAC_VAL = 0.20        # 20% of val files
SEARCH_EPOCHS_SUBSET = 40
N_TRIALS_SUBSET = 30

# =========================================================
# Stage 2 (full) settings
# =========================================================
SEARCH_EPOCHS_FULL = 100
TOP_K_FROM_SUBSET = 5        # take best K subset trials into full stage
N_EXTRA_FULL_TRIALS = 5       # optional extra exploration on full data

# =========================================================
# Fixed parameters (can tune later if desired)
# =========================================================
WEIGHT_DECAY = 1e-5
CONSISTENCY_WEIGHT = 0.1
HOLDOUT_FRAC = 0.10

# Hyperparameter search space
HIDDEN_SIZE_CHOICES = [16, 32, 64, 128]
BATCH_SIZE_CHOICES = [16, 32, 64, 128]
LR_LOW = 1e-4
LR_HIGH = 3e-3



def pick_fixed_subset(files, frac, seed):
    """
    Returns a fixed random subset of a list of files.
    """
    if frac >= 1.0:
        return list(files)

    n_total = len(files)
    n_keep = max(1, int(round(frac * n_total)))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_keep, replace=False)
    idx = np.sort(idx)

    return [files[i] for i in idx]


def build_loaders_from_files(batch_size, train_files, val_files):
    """
    Build train/val loaders for a given list of files.
    Normalization stats are computed from the training files only.
    """
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


def train_and_eval_one_epoch(model, optimizer, train_loader, val_loader, device):
    """
    Runs one training epoch + one validation pass.
    Returns mean validation MAE for that epoch.
    """
    model.train()

    for batch in train_loader:
        x = batch["x"].to(device)

        optimizer.zero_grad()
        loss, out, target_mask, metrics = training_step_loss(
            model,
            x,
            holdout_frac=HOLDOUT_FRAC
        )
        loss.backward()
        optimizer.step()

    model.eval()
    val_maes = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            res = evaluate_on_holdout(model, x, holdout_frac=HOLDOUT_FRAC)
            val_maes.append(res["mae"].item())

    return float(np.mean(val_maes))


def make_objective(train_files, val_files, search_epochs, stage_name):
    """
    Returns an Optuna objective bound to a specific file split and epoch budget.
    """
    def objective(trial: optuna.Trial) -> float:
        print(f"\n[{stage_name}] Starting trial {trial.number}")

        set_seed(SEED)

        hidden_size = trial.suggest_categorical("hidden_size", HIDDEN_SIZE_CHOICES)
        batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES)
        lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)

        train_loader, val_loader = build_loaders_from_files(
            batch_size=batch_size,
            train_files=train_files,
            val_files=val_files,
        )

        model = BRITSImputer(
            n_features=len(FEAT_COLS),
            hidden_size=hidden_size,
            consistency_weight=CONSISTENCY_WEIGHT,
        ).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )

        best_val_mae = float("inf")

        for epoch in range(search_epochs):
            mean_val_mae = train_and_eval_one_epoch(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=DEVICE,
            )

            best_val_mae = min(best_val_mae, mean_val_mae)

            print(
                f"[{stage_name}] Trial {trial.number:03d} | "
                f"Epoch {epoch+1:03d}/{search_epochs:03d} | "
                f"val_mae={mean_val_mae:.6f} | best={best_val_mae:.6f}"
            )

            trial.report(mean_val_mae, step=epoch)
            if trial.should_prune():
                print(f"[{stage_name}] Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()

        return best_val_mae

    return objective


def get_top_completed_trials(study, top_k=5):
    """
    Get the top completed trials sorted by objective value.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed = sorted(completed, key=lambda t: t.value)
    return completed[:top_k]


def save_study_summary(study, path):
    """
    Save a JSON summary of all completed trials.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    records = []
    for t in completed:
        records.append({
            "trial_number": t.number,
            "value": t.value,
            "params": t.params,
        })

    records = sorted(records, key=lambda x: x["value"])

    with open(path, "w") as f:
        json.dump(records, f, indent=2)


# =========================================================
# Main workflow
# =========================================================
def main():
    set_seed(SEED)

    # -----------------------------------------------------
    # 1) Create ONE fixed train/val split for all stages
    # -----------------------------------------------------
    all_files = list_psv_files(DATASET_DIR)
    train_files_full, val_files_full = split_files(
        all_files,
        train_frac=TRAIN_FRAC,
        seed=SEED
    )

    print(f"Total files: {len(all_files)}")
    print(f"Full train files: {len(train_files_full)}")
    print(f"Full val files:   {len(val_files_full)}")

    # -----------------------------------------------------
    # 2) Create ONE fixed subset split for stage 1
    # -----------------------------------------------------
    train_files_sub = pick_fixed_subset(
        train_files_full,
        frac=SUBSET_FRAC_TRAIN,
        seed=SEED + 100
    )

    val_files_sub = pick_fixed_subset(
        val_files_full,
        frac=SUBSET_FRAC_VAL,
        seed=SEED + 200
    )

    print(f"\nSubset train files: {len(train_files_sub)}")
    print(f"Subset val files:   {len(val_files_sub)}")

    # Save file counts and config for reproducibility
    config_summary = {
        "DATASET_DIR": DATASET_DIR,
        "OUT_DIR": OUT_DIR,
        "FEAT_COLS": FEAT_COLS,
        "T": T,
        "KEEP_LAST": KEEP_LAST,
        "NORMALIZE": NORMALIZE,
        "TRAIN_FRAC": TRAIN_FRAC,
        "SEED": SEED,
        "NUM_WORKERS": NUM_WORKERS,
        "SUBSET_FRAC_TRAIN": SUBSET_FRAC_TRAIN,
        "SUBSET_FRAC_VAL": SUBSET_FRAC_VAL,
        "SEARCH_EPOCHS_SUBSET": SEARCH_EPOCHS_SUBSET,
        "N_TRIALS_SUBSET": N_TRIALS_SUBSET,
        "SEARCH_EPOCHS_FULL": SEARCH_EPOCHS_FULL,
        "TOP_K_FROM_SUBSET": TOP_K_FROM_SUBSET,
        "N_EXTRA_FULL_TRIALS": N_EXTRA_FULL_TRIALS,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "CONSISTENCY_WEIGHT": CONSISTENCY_WEIGHT,
        "HOLDOUT_FRAC": HOLDOUT_FRAC,
        "DEVICE": str(DEVICE),
    }

    with open(os.path.join(OUT_DIR, "run_config.json"), "w") as f:
        json.dump(config_summary, f, indent=2)

    # -----------------------------------------------------
    # 3) Stage 1: Optuna on fixed subset
    # -----------------------------------------------------
    subset_study_name = "brits_imputation_subset_stage"
    subset_db_path = os.path.join(OUT_DIR, "subset_stage_optuna.db")

    subset_study = optuna.create_study(
        direction="minimize",
        study_name=subset_study_name,
        storage=f"sqlite:///{subset_db_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,
            n_warmup_steps=5
        ),
    )

    subset_objective = make_objective(
        train_files=train_files_sub,
        val_files=val_files_sub,
        search_epochs=SEARCH_EPOCHS_SUBSET,
        stage_name="SUBSET"
    )

    print("\n==============================")
    print("Running Stage 1: SUBSET TUNING")
    print("==============================")
    subset_study.optimize(subset_objective, n_trials=N_TRIALS_SUBSET)

    print("\nStage 1 best value:", subset_study.best_value)
    print("Stage 1 best params:")
    for k, v in subset_study.best_params.items():
        print(f"  {k}: {v}")

    with open(os.path.join(OUT_DIR, "best_params_subset.json"), "w") as f:
        json.dump(subset_study.best_params, f, indent=2)

    save_study_summary(
        subset_study,
        os.path.join(OUT_DIR, "subset_stage_all_completed_trials.json")
    )

    # -----------------------------------------------------
    # 4) Get top K subset trials
    # -----------------------------------------------------
    top_trials = get_top_completed_trials(subset_study, top_k=TOP_K_FROM_SUBSET)

    if len(top_trials) == 0:
        raise RuntimeError("No completed trials found in subset study.")

    print(f"\nTop {len(top_trials)} subset trials that will seed the full stage:")
    for rank, t in enumerate(top_trials, start=1):
        print(f"  Rank {rank}: value={t.value:.6f}, params={t.params}")

    top_trial_records = [
        {
            "rank": i + 1,
            "trial_number": t.number,
            "value": t.value,
            "params": t.params,
        }
        for i, t in enumerate(top_trials)
    ]

    with open(os.path.join(OUT_DIR, "top_subset_trials.json"), "w") as f:
        json.dump(top_trial_records, f, indent=2)

    # -----------------------------------------------------
    # 5) Stage 2: Full-data Optuna seeded by top subset trials
    # -----------------------------------------------------
    full_study_name = "brits_imputation_full_stage"
    full_db_path = os.path.join(OUT_DIR, "full_stage_optuna.db")

    full_study = optuna.create_study(
        direction="minimize",
        study_name=full_study_name,
        storage=f"sqlite:///{full_db_path}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10
        ),
    )

    # Seed the full study with the top subset params
    # This makes Optuna run these exact parameter sets first.
    for t in top_trials:
        full_study.enqueue_trial(t.params)

    full_objective = make_objective(
        train_files=train_files_full,
        val_files=val_files_full,
        search_epochs=SEARCH_EPOCHS_FULL,
        stage_name="FULL"
    )

    # Run the seeded top-K trials + optional extra trials
    n_full_trials_total = len(top_trials) + N_EXTRA_FULL_TRIALS

    print("\n====================================")
    print("Running Stage 2: FULL-DATA CONFIRMATION")
    print("====================================")
    print(f"Full stage total trials: {n_full_trials_total}")
    print(f"  - seeded from subset: {len(top_trials)}")
    print(f"  - extra full trials:  {N_EXTRA_FULL_TRIALS}")

    full_study.optimize(full_objective, n_trials=n_full_trials_total)

    print("\nStage 2 best value:", full_study.best_value)
    print("Stage 2 best params:")
    for k, v in full_study.best_params.items():
        print(f"  {k}: {v}")

    with open(os.path.join(OUT_DIR, "best_params_full.json"), "w") as f:
        json.dump(full_study.best_params, f, indent=2)

    save_study_summary(
        full_study,
        os.path.join(OUT_DIR, "full_stage_all_completed_trials.json")
    )

    # -----------------------------------------------------
    # 6) Final summary
    # -----------------------------------------------------
    final_summary = {
        "subset_best_value": subset_study.best_value,
        "subset_best_params": subset_study.best_params,
        "full_best_value": full_study.best_value,
        "full_best_params": full_study.best_params,
    }

    with open(os.path.join(OUT_DIR, "final_summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")
    print("Subset best value:", subset_study.best_value)
    print("Subset best params:", subset_study.best_params)
    print("Full best value:", full_study.best_value)
    print("Full best params:", full_study.best_params)


if __name__ == "__main__":
    main()
from libraries import *
import os
import json
import optuna

from train_utils import run_stratified_splits
from run_lstm import build_Xy
from plot_utils import make_all_summary_plots, save_all_open_figures

np.random.seed(1)
rn.seed(1)
tf.random.set_seed(1)

X, y, numTS, numVariables, meta = build_Xy(verbose=False)

NUM_EPOCHS = 50
N_SPLITS = 1
TEST_SIZE = 0.2
BASE_SEED = 123
OUT_DIR = os.path.join(meta["filesPath"], "LSTM_Tuning_All")

N_TRIALS = 150


def params_already_tested(study, current_params):
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        old_params = {}
        for k in current_params.keys():
            old_params[k] = t.params.get(k)

        if old_params == current_params:
            return True
    return False


def objective(trial):

    numNodes = trial.suggest_categorical("numNodes", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    percDropout = trial.suggest_float("percDropout", 0.10, 0.60, step=0.05)

    learning_rate = round( trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True), 6 )

    weight_decay = round( trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True), 8 )

    alpha = trial.suggest_float("alpha", 0.25, 0.95, step=0.05)
    gamma = trial.suggest_float("gamma", 1.0, 5.0, step=0.5)

    current_params = {
        "numNodes": numNodes,
        "batch_size": batch_size,
        "percDropout": percDropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "alpha": alpha,
        "gamma": gamma,
    }

    if params_already_tested(trial.study, current_params):
        raise optuna.exceptions.TrialPruned()

    results = run_stratified_splits(
        X=X,
        y=y,
        numTS=numTS,
        numVariables=numVariables,
        numNodes=numNodes,
        percDropout=percDropout,
        numEpochs=NUM_EPOCHS,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        base_seed=BASE_SEED,
        alpha=alpha,
        gamma=gamma,
        trial=trial,
        prune_metric="val_auprc",
        early_stopping_metric="val_auprc",
        early_stopping_patience=10,
        verbose_fit=2,
    )

    mean_ap = float(np.nanmean(results["all_ap"]))
    mean_auroc = float(np.nanmean(results["all_auroc"]))

    trial.set_user_attr("mean_auroc", mean_auroc)
    return mean_ap


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    sampler = optuna.samplers.TPESampler(
        seed=BASE_SEED,
        multivariate=True,
        n_startup_trials=10,
    )

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=8,
        max_resource=NUM_EPOCHS,
        reduction_factor=2,
    )

    study = optuna.create_study(
        direction="maximize",
        study_name="lstm_tuning_all_params",
        storage="sqlite:///lstm_tuning_all_params.db",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest AP:", study.best_value)
    print("Best params:", study.best_params)
    print("Best AUROC:", study.best_trial.user_attrs.get("mean_auroc"))

    best_params = study.best_params

    best_results = run_stratified_splits(
        X=X,
        y=y,
        numTS=numTS,
        numVariables=numVariables,
        numNodes=best_params["numNodes"],
        percDropout=best_params["percDropout"],
        numEpochs=NUM_EPOCHS,
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        base_seed=BASE_SEED,
        alpha=best_params["alpha"],
        gamma=best_params["gamma"],
        verbose_fit=2,
    )

    make_all_summary_plots(best_results, title_suffix=" (best trial rerun)")
    save_all_open_figures(
        out_dir=OUT_DIR,
        pdf_filename="best_trial_plots.pdf",
        dpi=300,
        close_after=True,
    )

    summary = {
        "best_ap": float(study.best_value),
        "best_auroc": float(study.best_trial.user_attrs.get("mean_auroc")),
        "best_params": best_params,
        "fixed_params": {
            "numEpochs": NUM_EPOCHS,
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "base_seed": BASE_SEED,
        },
        "optuna": {
            "sampler": "TPESampler(multivariate=True)",
            "pruner": "HyperbandPruner",
            "n_trials": N_TRIALS,
        },
    }

    with open(os.path.join(OUT_DIR, "best_trial_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved tuning summary to: {os.path.join(OUT_DIR, 'best_trial_summary.json')}")
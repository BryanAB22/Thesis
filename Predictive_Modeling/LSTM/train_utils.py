
import optuna
from libraries import *
from model_utils import create_model


class OptunaPruningCallback(tf.keras.callbacks.Callback):
    """Report validation metric to Optuna and prune weak trials early."""

    def __init__(self, trial, monitor="val_auprc", step_offset=0):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        if self.trial is None:
            return

        logs = logs or {}
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        step = self.step_offset + epoch + 1
        self.trial.report(float(current_value), step=step)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned at step={step} with {self.monitor}={float(current_value):.6f}"
            )


def set_seeds(seed):
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)


def run_stratified_splits(
    X,
    y,
    numTS,
    numVariables,
    numNodes,
    percDropout,
    numEpochs,
    batch_size,
    learning_rate,
    weight_decay,
    alpha,
    gamma,
    n_splits=5,
    test_size=0.2,
    base_seed=1,
    trial=None,
    prune_metric="val_auprc",
    early_stopping_metric="val_auprc",
    early_stopping_patience=10,
    verbose_fit=2,
):
    sss = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=base_seed,
    )

    all_acc = []
    all_auroc = []
    all_ap = []
    all_scores = []

    best_split_id = None
    best_ap = -np.inf
    best_history = None
    best_precision = None
    best_recall = None
    best_y_true = None
    best_y_proba = None
    best_y_pred = None
    best_fpr = None
    best_tpr = None
    best_auroc = None

    split_id = 0
    for train_idx, val_idx in sss.split(X, y):
        split_id += 1

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"\nSplit {split_id}/{n_splits}:")
        print("  Train patients:      {}".format(X_train.shape[0]))
        print("  Validation patients: {}".format(X_val.shape[0]))

        K.clear_session()
        set_seeds(base_seed + split_id)

        model = create_model(
            numTS,
            numVariables,
            numNodes,
            percDropout,
            learning_rate,
            weight_decay,
            alpha,
            gamma,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_metric,
                mode="max",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        ]

        if trial is not None:
            callbacks.append(
                OptunaPruningCallback(
                    trial=trial,
                    monitor=prune_metric,
                    step_offset=(split_id - 1) * numEpochs,
                )
            )

        fit_kwargs = dict(
            x=X_train,
            y=y_train,
            epochs=numEpochs,
            validation_data=(X_val, y_val),
            verbose=verbose_fit,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        history = model.fit(**fit_kwargs)

        scores = model.evaluate(X_val, y_val, verbose=0)
        all_scores.append(scores)

        y_val_proba = model.predict(X_val, batch_size=batch_size, verbose=0).ravel()
        y_val_pred = (y_val_proba >= 0.5).astype(int)

        val_acc = accuracy_score(y_val, y_val_pred)

        try:
            val_auroc = roc_auc_score(y_val, y_val_proba)
            fpr, tpr, _ = roc_curve(y_val, y_val_proba)
        except ValueError:
            val_auroc = np.nan
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])

        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        ap = average_precision_score(y_val, y_val_proba)

        all_acc.append(val_acc)
        all_auroc.append(val_auroc)
        all_ap.append(ap)

        print(
            "  evaluate -> loss={:.4f}, acc={:.4f}, auroc(keras)={:.4f}, auprc(keras)={:.4f}".format(
                scores[0], scores[1], scores[2], scores[3]
            )
        )
        print("  Accuracy = {:.4f}".format(val_acc))
        print("  AUROC    = {}".format("nan" if np.isnan(val_auroc) else f"{val_auroc:.4f}"))
        print("  AP (PR)  = {:.4f}".format(ap))
        print("  Positive rate = {:.4f}".format(y_val.mean()))

        if ap > best_ap:
            best_ap = ap
            best_split_id = split_id
            best_history = history
            best_precision = precision
            best_recall = recall
            best_y_true = y_val.copy()
            best_y_proba = y_val_proba.copy()
            best_y_pred = y_val_pred.copy()
            best_fpr = fpr.copy()
            best_tpr = tpr.copy()
            best_auroc = val_auroc

    all_scores = np.array(all_scores)

    results = {
        "all_scores": all_scores,
        "all_acc": np.array(all_acc),
        "all_auroc": np.array(all_auroc),
        "all_ap": np.array(all_ap),
        "best_split_id": best_split_id,
        "best_history": best_history,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_ap": best_ap,
        "best_y_true": best_y_true,
        "best_y_proba": best_y_proba,
        "best_y_pred": best_y_pred,
        "best_fpr": best_fpr,
        "best_tpr": best_tpr,
        "best_auroc": best_auroc,
    }
    return results

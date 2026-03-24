from libraries import *
# plot_utils.py


def _safe_name(text: str) -> str:
    text = (text or "figure").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "figure"


def plot_pr_curve(recall, precision, ap, title_suffix=""):
    fig = plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP = {ap:.3f}){title_suffix}")
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, auroc, title_suffix=""):
    fig = plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_loss(history, title_suffix=""):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig = plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Binary Focal Crossentropy Loss")
    plt.title(f"Loss vs Epochs{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_metric_history(history, metric_name="auprc", title_suffix=""):
    train_metric = history.history.get(metric_name, None)
    val_metric = history.history.get(f"val_{metric_name}", None)

    if train_metric is None or val_metric is None:
        print(f"[plot_metric_history] Metric '{metric_name}' not found in history.")
        return None

    epochs = range(1, len(train_metric) + 1)

    fig = plt.figure()
    plt.plot(epochs, train_metric, label=f"Train {metric_name.upper()}")
    plt.plot(epochs, val_metric, label=f"Validation {metric_name.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.upper())
    plt.title(f"{metric_name.upper()} vs Epochs{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, normalize=False, title_suffix=""):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix{title_suffix}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    return fig


def plot_threshold_curves(y_true, y_proba, title_suffix=""):
    thresholds = np.linspace(0.0, 1.0, 201)

    precisions = []
    recalls = []
    f1s = []

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    fig = plt.figure()
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision / Recall / F1 vs Threshold{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_probability_histogram(y_true, y_proba, bins=25, title_suffix=""):
    fig = plt.figure()
    plt.hist(y_proba[y_true == 0], bins=bins, alpha=0.7, label="True Class 0")
    plt.hist(y_proba[y_true == 1], bins=bins, alpha=0.7, label="True Class 1")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title(f"Predicted Probabilities by Class{title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return fig


def plot_split_boxplots(all_acc, all_auroc, all_ap, title_suffix=""):
    data = [all_acc, all_auroc, all_ap]
    labels = ["Accuracy", "AUROC", "AP"]

    fig = plt.figure()
    plt.boxplot(data, tick_labels=labels)
    plt.ylabel("Score")
    plt.title(f"Boxplots Across Splits{title_suffix}")
    plt.grid(True)
    plt.tight_layout()
    return fig


def save_all_open_figures(out_dir, pdf_filename="plots.pdf", dpi=300, close_after=False):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, pdf_filename)
    saved_pngs = []

    fig_nums = plt.get_fignums()
    if not fig_nums:
        print("[save_all_open_figures] No open figures to save.")
        return pdf_path, saved_pngs

    with PdfPages(pdf_path) as pdf:
        for idx, fig_num in enumerate(fig_nums, start=1):
            fig = plt.figure(fig_num)
            title = None
            axes = fig.get_axes()
            if axes:
                title = axes[0].get_title()
            png_name = f"{idx:02d}_{_safe_name(title)}.png"
            png_path = os.path.join(out_dir, png_name)
            fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            saved_pngs.append(png_path)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved {len(saved_pngs)} PNG files in: {out_dir}")

    if close_after:
        plt.close("all")

    return pdf_path, saved_pngs


def make_all_summary_plots(results, title_suffix=""):
    plot_pr_curve(results["best_recall"], results["best_precision"], results["best_ap"], title_suffix=title_suffix)
    plot_roc_curve(results["best_fpr"], results["best_tpr"], results["best_auroc"], title_suffix=title_suffix)
    plot_loss(results["best_history"], title_suffix=title_suffix)
    plot_metric_history(results["best_history"], metric_name="auprc", title_suffix=title_suffix)
    plot_metric_history(results["best_history"], metric_name="auroc", title_suffix=title_suffix)
    plot_confusion_matrix(results["best_y_true"], results["best_y_pred"], normalize=False, title_suffix=title_suffix)
    plot_threshold_curves(results["best_y_true"], results["best_y_proba"], title_suffix=title_suffix)
    plot_probability_histogram(results["best_y_true"], results["best_y_proba"], bins=25, title_suffix=title_suffix)
    plot_split_boxplots(results["all_acc"], results["all_auroc"], results["all_ap"], title_suffix=title_suffix)
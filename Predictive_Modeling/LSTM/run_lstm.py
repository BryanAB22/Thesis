from libraries import *
from train_utils import run_stratified_splits
from plot_utils import make_all_summary_plots, save_all_open_figures


def set_seeds(seed=1):
    np.random.seed(seed)
    rn.seed(seed)
    tf.random.set_seed(seed)


from data_utils import load_csvs, getTrainData


def build_Xy(
    filesPath="/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/",
    numTSInitial=48,
    numHRBef=6,
    ignore=("SepsisLabel",),
    verbose=True,
):
    if verbose:
        print('\n************** Classification of Sepsis using LSTM Model **************')
        print("\nDate and time =", datetime.now().strftime("%m/%d/%Y %H:%M:%S"))

    ignore = list(ignore)
    pacsPositives, pacsNegatives = load_csvs(filesPath, drop_cols=ignore)

    columns = pacsPositives.columns

    numPacsPos = pacsPositives.shape[0] // numTSInitial
    numPacsNeg = pacsNegatives.shape[0] // numTSInitial

    if verbose:
        print('%i patients (%i positives, %i negatives)' % (numPacsPos + numPacsNeg, numPacsPos, numPacsNeg))

    pacsTrain, y = getTrainData(pacsPositives, pacsNegatives, numTSInitial)
    lenTrain = np.shape(pacsTrain)

    if verbose:
        print("Total patients in dataset: {}".format(lenTrain[0]))

    numTS = numTSInitial - numHRBef

    if verbose:
        print('\nCLASSIFICATION AT %i HOURS BEFORE SEPSIS ONSET (%i TIMESTAMPS).' % (numHRBef, numTS))

    indsImportVars = list(range(len(columns)))
    numVariables = len(indsImportVars)

    X = pacsTrain[:, 0:numTS, indsImportVars]

    if verbose:
        print('\n*******************************************************************')

    meta = {
        "columns": columns,
        "numPacsPos": numPacsPos,
        "numPacsNeg": numPacsNeg,
        "numTSInitial": numTSInitial,
        "numHRBef": numHRBef,
        "filesPath": filesPath,
    }

    return X, y, numTS, numVariables, meta


if __name__ == "__main__":
    set_seeds(1)

    numNodes = 64
    percDropout = 0.11839344880449137
    numEpochs = 200
    learning_rate = 0.0002829898808466607
    batch_size = 32
    weight_decay = 0.00038036792991304555
    class_weights = True
    alpha=.9
    gamma=1
    n_splits = 10
    test_size = 0.2

    numTSInitial = 48
    numHRBef = 6

    filesPath = '/Users/bryanbarrios/Desktop/MastersResearch/PredictiveModeling/Data/'

    X, y, numTS, numVariables, meta = build_Xy(
        filesPath=filesPath,
        numTSInitial=numTSInitial,
        numHRBef=numHRBef,
        ignore=("SepsisLabel",),
        verbose=True
    )

    print("Model with %i Nodes, %.2f Dropout, %i Epochs and %i Batch Size." % (numNodes, percDropout, numEpochs, batch_size))

    initialTime = timeit.default_timer()
    print(f"\nTraining LSTM model with StratifiedShuffleSplit (n_splits={n_splits})...")

    results = run_stratified_splits(
        X=X,
        y=y,
        numTS=numTS,
        numVariables=numVariables,
        numNodes=numNodes,
        percDropout=percDropout,
        numEpochs=numEpochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        alpha=alpha,
        gamma=gamma,
        n_splits=n_splits,
        test_size=test_size,
        base_seed=1,
    )

    all_scores = results["all_scores"]
    all_acc = results["all_acc"]
    all_auroc = results["all_auroc"]
    all_ap = results["all_ap"]

    print("\n================ Summary over splits ================")
    print("Loss:     mean={:.4f} std={:.4f}".format(all_scores[:, 0].mean(), all_scores[:, 0].std()))
    print("Accuracy: mean={:.4f} std={:.4f}".format(all_acc.mean(), all_acc.std()))
    print("AUROC:    mean={:.4f} std={:.4f}".format(np.nanmean(all_auroc), np.nanstd(all_auroc)))
    print("AP (PR):  mean={:.4f} std={:.4f}".format(all_ap.mean(), all_ap.std()))

    make_all_summary_plots(results, title_suffix=" (best AP split)")

    out_dir = os.path.join(meta["filesPath"], "LSTM_Plots")
    save_all_open_figures(out_dir=out_dir, pdf_filename="run_lstm_plots.pdf", dpi=300, close_after=False)

    elapsed = timeit.default_timer() - initialTime
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")
    plt.show()

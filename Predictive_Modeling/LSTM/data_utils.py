
from libraries import *
# data_utils.py

def getSequences(PacsPositives, PacsNegatives, numTS):
    xPacsPositives = PacsPositives.values[:, :]
    xPacsNegatives = PacsNegatives.values[:, :]

    sequencesPos = []
    for i in range(0, len(xPacsPositives), numTS):
        sequencesPos.append(xPacsPositives[i:i + numTS, :])

    sequencesNeg = []
    for i in range(0, len(xPacsNegatives), numTS):
        sequencesNeg.append(xPacsNegatives[i:i + numTS, :])

    return sequencesPos, sequencesNeg


def getTrainData(PacsPositives, PacsNegatives, numTSInitial):
    sequencesPos, sequencesNeg = getSequences(PacsPositives, PacsNegatives, numTSInitial)

    lenTrainPos = len(sequencesPos)
    lenTrainNeg = len(sequencesNeg)

    train = sequencesPos[0:lenTrainPos] + sequencesNeg[0:lenTrainNeg]
    train_target = np.concatenate((np.repeat(1, lenTrainPos), np.repeat(0, lenTrainNeg)))

    train = np.array(train)

    inds = np.random.choice(len(train), len(train), replace=False)
    train = train[inds]
    train_target = train_target[inds]

    return train, train_target


def load_csvs(filesPath, drop_cols=None):
    pacsPositives = pd.read_csv(filesPath + 'septic.csv')
    pacsNegatives = pd.read_csv(filesPath + 'nonseptic.csv')
    
    if drop_cols:
        pacsPositives = pacsPositives.drop(columns=drop_cols, errors='ignore')
        pacsNegatives = pacsNegatives.drop(columns=drop_cols, errors='ignore')

        common = pacsPositives.columns.intersection(pacsNegatives.columns)
        pacsPositives = pacsPositives[common]
        pacsNegatives = pacsNegatives[common]
    return pacsPositives, pacsNegatives

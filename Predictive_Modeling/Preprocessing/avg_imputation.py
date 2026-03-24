from libraries import *

ignore = {"SepsisLabel", "Unit1", "Unit2"}
main_dir=main_dir
os.makedirs(avg_dir, exist_ok=True)
for fn in glob.glob(os.path.join(main_dir, "*.psv")):
    df = pd.read_csv(fn, sep="|", na_values=[""], keep_default_na=True)
    numCols = [
        c for c in df.columns
        if c not in ignore and np.issubdtype(df[c].dtype, np.number)
    ]
    

    for col in feat:
        arr = df[col].to_numpy()
        originalIndex = np.where(~np.isnan(arr))[0]
        for i, j in zip(originalIndex, originalIndex[1:]):
            if j - i == 2:
                arr[i+1] = (arr[i] + arr[j]) / 2.0
        df[col] = arr

    outputPath = os.path.join(avg_dir, os.path.basename(fn))
    df.to_csv(outputPath, sep="|", index=False)
import pandas as pd
from import_data import import_from_files, get_filenames, get_tablename
from read_data import read_from_db

def _main() -> None:
    filenames = get_filenames()
    for filename in filenames:
        tablename = get_tablename(filename)
        df = read_from_db(tablename=tablename)
        if df is None:
            print("Error")
            continue
        preprocess(df)

def preprocess(df: pd.DataFrame) -> None:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    dt = df.index.to_series().diff().median().total_seconds()
    print("[s]", dt)
    
def main() -> None:
    import_from_files()
    _main()

if __name__ == "__main__":
    main()
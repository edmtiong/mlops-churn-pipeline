import pandas as pd
from pathlib import Path

#Path resolves to project root
RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load_and_preprocess(path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw Telco churn CSV and return clean features X and target y.
    """
    df = pd.read_csv(path)

    # Fix 1: TotalCharges is string — blank strings exist for tenure=0 customers
    # pd.to_numeric with errors='coerce' converts blanks to NaN, then we fill with 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Fix 2: customerID is an identifier, not a feature
    df = df.drop(columns=["customerID"])

    # Fix 3: Churn is Yes/No string — convert to binary 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Fix 4: Encode remaining string columns as integer category codes
    # e.g. 'Male'/'Female' becomes 0/1
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = pd.Categorical(df[col]).codes

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess()
    print(f"Features shape: {X.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"\nTarget distribution:\n{y.value_counts()}")
    print(f"\nData types after preprocessing:\n{X.dtypes}")
    print(f"\nMissing values after preprocessing:\n{X.isnull().sum().sum()} total")
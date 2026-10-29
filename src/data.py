from pathlib import Path
import pandas as pd
from src.config import load_params

def download(p: dict) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(p["data"]["hf_dataset"])
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split].to_pandas()

def validate(df: pd.DataFrame, target: str) -> None:
    """Contrôles de qualité minimaux (remplaçables par Pandera / Great Expectations)."""
    assert target in df.columns, f"Colonne cible absente: {target}"
    assert df[target].notna().all(), "Valeurs nulles dans la cible"
    assert len(df) > 0, "Jeu de données vide"
    if "tenure_months" in df.columns:
        assert (df["tenure_months"].dropna() >= 0).all(), "tenure_months négatif"

def load_data(p: dict) -> pd.DataFrame:
    raw = Path(p["data"]["raw_path"])
    if raw.exists():
        df = pd.read_csv(raw)
    else:
        df = download(p)
        raw.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw, index=False)
    validate(df, p["data"]["target"])
    return df

if __name__ == "__main__":
    d = load_data(load_params())
    print(d.shape)
    print(d.dtypes)

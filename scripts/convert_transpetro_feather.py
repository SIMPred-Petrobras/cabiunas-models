"""Converte os .feather da Transpetro (índice datetime, grade regular) para CSV com
coluna de tempo explícita, no formato que io.py::load_data espera (pd.read_csv +
cfg.TIME_COL).

Uso:
  python scripts/convert_transpetro_feather.py
"""
from pathlib import Path
import pandas as pd

SRC_DIR = Path("../dados/drive-download-20260720T190351Z-1-001")
OUT_DIR = Path("../dados/transpetro")
FILES = ["B-4064A.feather", "B-90001A.feather", "B-8802B.feather", "B-402E.feather"]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        df = pd.read_feather(SRC_DIR / name)
        if "Data Hora" in df.columns:
            # B-402E: já vem com coluna de tempo explícita e índice = RangeIndex sem
            # significado (resetar o índice geraria um "data_datetime" de inteiros 0..N).
            df = df.rename(columns={"Data Hora": "data_datetime"})
        else:
            df = df.reset_index().rename(columns={"index": "data_datetime"})
        out = OUT_DIR / (Path(name).stem + "_1min.csv")
        df.to_csv(out, index=False)
        print(f"{name}: {df.shape} -> {out} "
              f"({df['data_datetime'].min()} .. {df['data_datetime'].max()})")


if __name__ == "__main__":
    main()

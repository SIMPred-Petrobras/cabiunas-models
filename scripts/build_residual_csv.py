"""Gera o CSV de resíduo common-mode para o experimento v11.

Para cada termopar TC382_0X: residual = TC_X − média dos outros 5 TCs válidos.
Para T5_AVG: residual = T5 − média dos 6 TCs válidos (termopar − perfil da turbina).
"Válido" = valor em [500, 950] (mesma faixa do SENTINEL do v9/v10) e RUNNING_A > 0.5,
com ≥3 pares válidos. Fora disso o resíduo é exatamente 0.0 — runs constantes de 0
são excluídos do treino por EXCLUDE_CONSTANT_RUNS, replicando o efeito do sentinela.

Motivação: carga/ambiente/queima movem os 7 sensores juntos (common-mode). O resíduo
desconta isso por construção — alvo: TC03 backcast-2024 >21% e TC05 >50% de recall_raw
sem perder o 17/17 OOS (memória t5-drift-common-mode, experimento ctx 17.62x).

Uso:
  python scripts/build_residual_csv.py
"""
import numpy as np
import pandas as pd

SRC = "../dados/sensores_2024h2_2025_2026_30s.csv"
OUT = "../dados/sensores_residual_2024h2_2025_2026_30s.csv"
TCS = ["TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A", "TC382_05_A", "TC382_06_A"]
LOW, HIGH = 500.0, 950.0
MIN_PEERS = 3


def main():
    df = pd.read_csv(SRC, low_memory=False)
    t = pd.to_datetime(df["data_datetime"], utc=True, errors="coerce")
    df = df.loc[t.notna()].copy()
    df["data_datetime"] = t[t.notna()].values

    running = pd.to_numeric(df["RUNNING_A"], errors="coerce").fillna(0.0)
    on = (running > 0.5).to_numpy()

    masked = {}
    for c in TCS + ["T5_AVG_A"]:
        v = pd.to_numeric(df[c], errors="coerce")
        masked[c] = v.where((v >= LOW) & (v <= HIGH))

    tc_mat = pd.concat([masked[c] for c in TCS], axis=1)
    tc_mat.columns = TCS
    n_valid = tc_mat.notna().sum(axis=1)

    out = pd.DataFrame({"data_datetime": df["data_datetime"].values,
                        "RUNNING_A": running.values})
    for c in TCS:
        peers = [p for p in TCS if p != c]
        peer_mean = tc_mat[peers].mean(axis=1)
        n_peers = tc_mat[peers].notna().sum(axis=1)
        resid = masked[c] - peer_mean
        ok = on & masked[c].notna().to_numpy() & (n_peers >= MIN_PEERS).to_numpy()
        out[c] = np.where(ok, resid.to_numpy(), 0.0)
    t5_mean = tc_mat.mean(axis=1)
    resid5 = masked["T5_AVG_A"] - t5_mean
    ok5 = on & masked["T5_AVG_A"].notna().to_numpy() & (n_valid >= MIN_PEERS).to_numpy()
    out["T5_AVG_A"] = np.where(ok5, resid5.to_numpy(), 0.0)

    out.to_csv(OUT, index=False)
    print(f"linhas: {len(out)}  ON: {on.mean()*100:.1f}%")
    for c in ["T5_AVG_A"] + TCS:
        nz = out.loc[out[c] != 0.0, c]
        print(f"  {c}: nao-zero {len(nz)/len(out)*100:5.1f}%  p50={nz.median():7.2f}  "
              f"p05={nz.quantile(.05):7.2f}  p95={nz.quantile(.95):7.2f}")
    print(f"csv: {OUT}")


if __name__ == "__main__":
    main()

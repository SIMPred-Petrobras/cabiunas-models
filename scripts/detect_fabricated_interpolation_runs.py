"""Detecta trechos suspeitos de interpolação linear "fabricada" (sem base em dado real do
PI) no dataset completo `sensores_filtrados_Interpolados_2025.csv` (dataset ClearML
e2765c3eef2349cda5f5cbcb0fcd5a40), sem precisar de novas consultas ao histórico bruto.

Assinatura usada (validada contra a Janela 6 de TC382_03_A_recorded_janelas_2025.xlsx, o
único caso confirmado com dado real de verdade pra comparar): um trecho de interpolação
linear tem curvatura (segunda diferença discreta, diff.diff()) quase zero, sustentada por
muitos pontos seguidos — muito abaixo do que um sinal real e ruidoso produz. Real: diff2
tipicamente da ordem de unidades (o sensor oscila). Fabricado (Janela 6 confirmada): diff2
< 0.02 por 881 pontos (7h20) seguidos.

Regra: marca como "suspeito" qualquer ponto com |diff2| < EPS, e reporta só os RUNS com
duração >= MIN_RUN_MINUTES minutos. Cruza com RUNNING_A pra separar:
  - runs com equipamento LIGADO -> alto risco (mesmo padrão da Janela 6 confirmada)
  - runs com equipamento DESLIGADO -> provável compressão normal do PI (benigno, como
    Janela 1/4 do teste manual)

Uso:
  PYTHONPATH=. python scripts/detect_fabricated_interpolation_runs.py [SENSOR]
  (default: TC382_03_A)
"""
import sys
import numpy as np
import pandas as pd

INTERP_CSV = "/home/thallys/.clearml/cache/storage_manager/datasets/ds_e2765c3eef2349cda5f5cbcb0fcd5a40/sensores_filtrados_Interpolados_2025.csv"
EPS = 0.02          # curvatura abaixo disso = suspeito (Janela 6 confirmada tinha <0.02)
MIN_RUN_MINUTES = 30.0
DT_SECONDS = 30.0


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def main():
    sensor = sys.argv[1] if len(sys.argv) > 1 else "TC382_03_A"
    print(f"Carregando {sensor}...")
    df = pd.read_csv(INTERP_CSV, usecols=["data_datetime", sensor, "RUNNING_A"])
    df["data_datetime"] = pd.to_datetime(df["data_datetime"], errors="coerce")
    df = df.dropna(subset=["data_datetime"]).sort_values("data_datetime").reset_index(drop=True)
    v = df[sensor].to_numpy(dtype=float)
    # RUNNING_A nao e binario limpo (0/1): tem valores como 5.0, 11.0 e fracoes
    # intermediarias (interpolacao/artefato na propria coluna de estado operacional) —
    # so 1.0 exato e "ligado" confirmado (validado contra a Janela 6, que tinha RUNNING_A=1.0).
    running = np.isclose(df["RUNNING_A"].to_numpy(dtype=float), 1.0, atol=1e-6)
    t = df["data_datetime"]

    diff2 = np.diff(v, n=2)
    diff2 = np.concatenate([[np.nan, np.nan], diff2])
    suspect = np.abs(diff2) < EPS
    suspect[:2] = False

    runs = find_runs(suspect)
    min_run_pts = int(MIN_RUN_MINUTES * 60 / DT_SECONDS)
    rows = []
    for (i0, i1) in runs:
        n_pts = i1 - i0 + 1
        if n_pts < min_run_pts:
            continue
        dur_min = n_pts * DT_SECONDS / 60.0
        frac_running = running[i0:i1 + 1].mean()
        rows.append(dict(
            inicio=t.iloc[i0], fim=t.iloc[i1], duracao_min=dur_min,
            v_inicio=round(v[i0], 2), v_fim=round(v[i1], 2),
            frac_equip_ligado=round(frac_running, 2),
        ))

    out = pd.DataFrame(rows).sort_values("duracao_min", ascending=False)
    out_csv = f"eda_load_residual_out/tc382_03_a_gap_check/suspeitos_{sensor}.csv"
    out.to_csv(out_csv, index=False)

    print(f"\n{len(out)} trechos suspeitos (>= {MIN_RUN_MINUTES:.0f}min, curvatura < {EPS}) em {sensor}")
    total_h = out["duracao_min"].sum() / 60.0
    ligado = out[out["frac_equip_ligado"] >= 0.8]
    desligado = out[out["frac_equip_ligado"] < 0.2]
    misto = out[(out["frac_equip_ligado"] >= 0.2) & (out["frac_equip_ligado"] < 0.8)]
    print(f"tempo total suspeito: {total_h:.1f}h de {(t.iloc[-1]-t.iloc[0]).total_seconds()/3600:.0f}h no periodo")
    print(f"  com equipamento LIGADO (risco alto, mesmo padrao da Janela 6 confirmada): "
         f"{len(ligado)} trechos, {ligado['duracao_min'].sum()/60:.1f}h")
    print(f"  com equipamento DESLIGADO (provavel compressao normal, benigno): "
         f"{len(desligado)} trechos, {desligado['duracao_min'].sum()/60:.1f}h")
    print(f"  misto (transicao liga/desliga): {len(misto)} trechos, {misto['duracao_min'].sum()/60:.1f}h")

    print(f"\nTop 15 trechos por duracao:")
    pd.set_option("display.width", 200)
    print(out.head(15).to_string(index=False))
    print(f"\ncsv completo: {out_csv}")


if __name__ == "__main__":
    main()

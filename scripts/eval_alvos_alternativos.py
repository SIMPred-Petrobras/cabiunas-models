#!/usr/bin/env python3
"""
eval_alvos_alternativos.py
Reavalia os modelos JÁ TREINADOS contra alvos que NÃO são função da entrada.

O problema com a métrica de sempre: os incidentes eram alarmes HI/HIHI do próprio
TC382_03_A, ou seja cruzamentos de limiar sobre o sinal que alimenta o modelo. O
alvo era função direta da entrada, e por isso uma EWMA da temperatura crua iguala
o autoencoder — o protocolo não conseguia demonstrar valor de modelagem.

A base de alarmes cobre a turbina inteira, não só os termopares. Daí saem alvos
de outra natureza física, que nenhum limiar sobre a temperatura de exaustão pode
produzir por construção:

    pressao   PAL_6240315 / PDAL_6240302 / PAH_6240319
              pressão de gás combustível e de selagem — 133 incidentes
    mancal    TAH_*/TAHH_* 6240301/03/05/07
              temperatura dos mancais radiais e de escora — 8 incidentes
    trip      PALL_6240340 / TALL_6240325 / PALL_6240309
              trips de óleo lubrificante — 2 incidentes com máquina rodando antes

Filtro de incidente, igual em espírito ao do protocolo original: só ativação
(Status ACT), só evento com a turbina comprovadamente rodando nas 2h anteriores
(o TRIP dispara depois de derrubar a máquina, então exigir ON no instante do
alarme descartaria todos), e agrupamento por GAP_HOURS.

Nenhum treino: reusa o `sequence_scores_all.csv` dos braços existentes e a
temperatura crua como linha de base.

Uso:
    PYTHONPATH=. python scripts/eval_alvos_alternativos.py
    PYTHONPATH=. python scripts/eval_alvos_alternativos.py --alvo mancal
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")

HL_GRID = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
N_Q = 120
GAP_H = 8.0
ARRAY = [f"TC382_0{i}_A" for i in range(1, 7)]

ALVOS = {
    "pressao": ["PAL_6240315", "PDAL_6240302", "PAH_6240319"],
    "mancal": ["TAH_6240301", "TAH_6240303", "TAH_6240305", "TAH_6240307",
               "TAHH_6240303", "TAHH_6240305", "TAHH_6240307"],
    "trip": ["PALL_6240340", "TALL_6240325", "PALL_6240309"],
}

ARMS = {
    "baseline_temp": "",
    "v16_excl_24h": "0fdeb5318361420e904b7994a65e3593",
    "v16_ctrl": "a78df1cf1c6e4f43ac9e1303034d71eb",
    "v18_residuo": "bf3d61e9f6a04e68ac9af255ec40ecbe",
}
GRADE = "0fdeb5318361420e904b7994a65e3593"


def _dados() -> str:
    for up in ("..", "../..", "../../.."):
        c = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(c):
            return c
    raise SystemExit("diretório 'dados/' não encontrado.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alvo", default=None, choices=list(ALVOS),
                   help="omitido = avalia todos")
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--horizonte", type=float, default=8.0)
    p.add_argument("--out", default="eval_predictive_out/alvos_alternativos.csv")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    S = a.sensor
    sw.SENSOR = S
    sw.HORIZON = a.horizonte
    D = _dados()
    sw.RAW_CSV = os.path.join(D, "sensores_2024h2_2025_2026_30s.csv")
    sw.ALARM_CSV = os.path.join(D, "alarmes_selecionados_turbina_a.csv")
    ev.ALARM_CSV_DEFAULT = sw.ALARM_CSV

    from clearml import Task

    raw = pd.read_csv(sw.RAW_CSV, usecols=["data_datetime", "RUNNING_A", *ARRAY],
                      low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    T = raw[ARRAY].apply(pd.to_numeric, errors="coerce")
    # "rodando" = flag ligada E array quente: o RUNNING_A fica em 1 com a turbina
    # fria (1.5% do tempo ON), e ali qualquer score vira constante
    running = ((pd.to_numeric(raw["RUNNING_A"], errors="coerce") > 0.5)
               & (T.mean(axis=1) >= 500.0)).astype(float)
    t5 = T.mean(axis=1)
    t5s = t5.ewm(halflife=int(pd.Timedelta(hours=24) / pd.Timedelta("30s"))).mean()

    al = pd.read_csv(sw.ALARM_CSV)
    al["t"] = pd.to_datetime(al["Data da Ocorrência"], errors="coerce", utc=True)
    al = al.dropna(subset=["t"])
    al = al[al["Status"].astype(str).str.startswith("ACT")]

    grade = sw.read_mae(Task.get_task(task_id=GRADE)
                        .artifacts[f"{S}_csv_sequence_scores_all.csv"].get_local_copy()).index
    on_bool = running > 0.5

    def incidentes(tags: list[str]) -> list:
        s = al[al["Tag Alarme"].isin(tags)]
        s = s[(s.t >= grade[0]) & (s.t <= grade[-1])]
        # a turbina tem de estar rodando em ALGUM momento das 2h anteriores: o
        # TRIP dispara depois de derrubar a máquina
        ok = [t for t in s.t
              if on_bool.loc[t - pd.Timedelta(hours=2): t].any()]
        if not ok:
            return []
        return ev.cluster_incidents(sorted(ok), gap_hours=GAP_H)

    def score_de(tid: str) -> pd.Series:
        if tid == "":
            b = T[S].reindex(grade, method="nearest")
            b = b.where(b >= 500).ffill().dropna()
            return b.rolling(window=10, min_periods=1).mean()
        return sw.read_mae(Task.get_task(task_id=tid)
                           .artifacts[f"{S}_csv_sequence_scores_all.csv"].get_local_copy())

    alvos = [a.alvo] if a.alvo else list(ALVOS)
    hs = sw.HORIZON * 3600.0
    linhas = []
    for nome in alvos:
        inc = incidentes(ALVOS[nome])
        print(f"\n=== alvo '{nome}': {len(inc)} incidentes agrupados a {GAP_H:g}h")
        if len(inc) < 3:
            print("   poucos incidentes para uma curva — pulando.")
            continue
        inc_s = np.array([t.timestamp() for t in inc])
        for lab, tid in ARMS.items():
            score = score_de(tid)
            dias = (score.index[-1] - score.index[0]).total_seconds() / 86400.0
            melhor = None
            for hl in HL_GRID:
                health = sw.health_global(score, hl, running, t5s)
                if health.empty:
                    continue
                for q in np.linspace(0.50, 0.999, N_Q):
                    if float((health >= q).mean()) > sw.MAX_DUTY:
                        continue
                    alerta = ev.apply_sticky(health, q, sw.STICKY)
                    duty = float(alerta.mean())
                    if duty > sw.MAX_STICKY:
                        continue
                    eps = ev.detect_episodes_gap(alerta)
                    cruz = np.array([t.timestamp() for t in health.index[health >= q]])
                    n_hit = sum(1 for ti in inc_s if cruz.size
                                and np.any((cruz >= ti - hs) & (cruz <= ti)))
                    n_fp = sum(1 for (x, y) in eps
                               if not np.any((inc_s - hs <= y.timestamp())
                                             & (inc_s >= x.timestamp())))
                    if n_fp / max(dias, 1.0) > sw.FA_BUDGET:
                        continue
                    cand = (n_hit / len(inc_s), -n_fp)
                    if melhor is None or cand > melhor[0]:
                        melhor = (cand, dict(alvo=nome, braco=lab, hl=hl,
                                             q=round(float(q), 4), inc=len(inc),
                                             recall=n_hit / len(inc_s), fp=n_fp,
                                             duty=duty, dias=round(dias)))
            if melhor:
                r = melhor[1]
                linhas.append(r)
                print(f"   {lab:16s} recall={r['recall']:6.1%}  FP={r['fp']:>4}  "
                      f"hl={r['hl']:<5} duty={r['duty']:.3f}")

    if not linhas:
        raise SystemExit("nenhum alvo avaliável.")
    df = pd.DataFrame(linhas)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\ngravado: {a.out}")


if __name__ == "__main__":
    main()

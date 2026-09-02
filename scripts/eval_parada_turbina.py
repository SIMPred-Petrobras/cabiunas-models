#!/usr/bin/env python3
"""
eval_parada_turbina.py
Avalia os modelos contra PARADA DA TURBINA, e não contra alarmes do próprio sensor.

Por que este alvo. Os alarmes HI/HIHI do TC382_03_A são cruzamentos de limiar
sobre o sinal que alimenta o modelo — alvo função direta da entrada, que uma EWMA
da temperatura crua prevê tão bem quanto o autoencoder. Pior: eles são seguidos de
parada em 8% dos casos, contra 6% de taxa base, ou seja praticamente não indicam
nada. Parada de máquina não é limiar de nenhum canal, tem contagem suficiente e
ninguém discute se importa.

Duas armadilhas tratadas explicitamente:

  RAMPA   se o alerta só sobe porque o desligamento já começou, não é previsão.
          O lead é medido até o INÍCIO do episódio de alerta, e o script reporta
          a fração de episódios que começam com a máquina ainda em temperatura
          plena, além da fração com pelo menos 6h de antecedência.

  NULO    "15h é menos que as 48h esperadas" é raciocínio, não medida: os
          episódios de alerta são longos e concentrados nos períodos de operação.
          Aqui a distribuição nula vem de permutação — paradas falsas sorteadas
          nos mesmos instantes elegíveis, mesma quantidade.

O RUNNING_A fica ligado com a turbina fria em ~1.5% do tempo, e ali qualquer score
vira constante; por isso "rodando" exige também o array acima de 500°C.

Uso:
    PYTHONPATH=. python scripts/eval_parada_turbina.py
    PYTHONPATH=. python scripts/eval_parada_turbina.py --arm v19=<task_id> --min_bloco 6
"""
from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ARRAY = [f"TC382_0{i}_A" for i in range(1, 7)]
HL_PADRAO = {"baseline_temp": 24.0}
N_PERM = 2000


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_per_sensor_level")
sw = _load("sweep_regime_band_offline")

ARMS = {
    "baseline_temp": "",
    "v16_ctrl": "a78df1cf1c6e4f43ac9e1303034d71eb",
    "v16_excl_24h": "0fdeb5318361420e904b7994a65e3593",
}


def _dados() -> str:
    for up in ("..", "../..", "../../.."):
        c = os.path.normpath(os.path.join(_HERE, "..", up, "dados"))
        if os.path.isdir(c):
            return c
    raise SystemExit("diretório 'dados/' não encontrado.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="sensores_consolidado_2022_2026_30s.csv")
    p.add_argument("--sensor", default="TC382_03_A")
    p.add_argument("--arm", action="append", default=[], help="rotulo=task_id (repetível)")
    p.add_argument("--hl", type=float, default=0.5, help="half-life dos braços de modelo")
    p.add_argument("--q", type=float, default=0.90, help="quantil do corte (duty comum)")
    p.add_argument("--min_bloco", type=float, default=6.0,
                   help="operação mínima antes da parada, em horas")
    p.add_argument("--janela", type=float, default=96.0, help="busca do alerta, em horas")
    p.add_argument("--desde", default=None,
                   help="corta score, incidentes e indice a partir desta data. Necessario\n"
                        "quando o modelo foi treinado depois de uma reconfiguracao do\n"
                        "instrumento: pontuar o periodo antigo satura o indice e, como ele\n"
                        "e rank percentual GLOBAL, empurra o corte e apaga o alerta no\n"
                        "periodo bom. No TC382 o array mudou em 2023.")
    p.add_argument("--perm", type=int, default=N_PERM)
    p.add_argument("--out", default="eval_predictive_out/parada_turbina.csv")
    return p.parse_args()


def blocos_quentes(on: pd.Series):
    v = on.to_numpy()
    idx = on.index
    corte = np.flatnonzero(v[1:] != v[:-1]) + 1
    ini = np.concatenate(([0], corte))
    fim = np.concatenate((corte, [len(v)]))
    return [(idx[a], idx[min(b, len(idx) - 1)]) for a, b in zip(ini, fim) if v[a]]


def inicio_episodio(cruz_s: np.ndarray, t_s: float, janela_s: float):
    w = cruz_s[(cruz_s <= t_s) & (cruz_s >= t_s - janela_s)]
    if not w.size:
        return None
    g = np.diff(w)
    k = np.flatnonzero(g > 3600.0)
    return w[k[-1] + 1] if k.size else w[0]


def main() -> None:
    a = parse_args()
    S = a.sensor
    D = _dados()
    sw.RAW_CSV = os.path.join(D, a.csv)
    sw.ALARM_CSV = os.path.join(D, "alarmes_selecionados_turbina_a.csv")
    ev.ALARM_CSV_DEFAULT = sw.ALARM_CSV
    sw.SENSOR = S

    from clearml import Task

    raw = pd.read_csv(sw.RAW_CSV, usecols=["data_datetime", "RUNNING_A", *ARRAY],
                      low_memory=False)
    raw["data_datetime"] = pd.to_datetime(raw["data_datetime"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["data_datetime"]).set_index("data_datetime").sort_index()
    T = raw[ARRAY].apply(pd.to_numeric, errors="coerce")
    on = (pd.to_numeric(raw["RUNNING_A"], errors="coerce") > 0.5) & (T.mean(axis=1) >= 500.0)
    running = on.astype(float)
    t5s = T.mean(axis=1).ewm(halflife=int(pd.Timedelta(hours=24) / pd.Timedelta("30s"))).mean()

    blocos = blocos_quentes(on)
    paradas = [f for i, f in blocos
               if (f - i).total_seconds() / 3600.0 >= a.min_bloco]
    print(f"base {a.csv}: {on.sum() * 30 / 3600:,.0f}h quentes, "
          f"{len(blocos)} blocos, {len(paradas)} paradas após ≥{a.min_bloco:g}h\n")

    arms = dict(ARMS)
    for spec in a.arm:
        k, v = spec.split("=", 1)
        arms[k.strip()] = v.strip()

    rng = np.random.default_rng(7)
    janela_s = a.janela * 3600.0
    linhas = []
    for lab, tid in arms.items():
        if tid == "":
            b = T[S].where(T[S] >= 500).ffill().dropna()
            score = b.rolling(window=10, min_periods=1).mean()
        else:
            art = f"{S}_csv_sequence_scores_all.csv"
            t = Task.get_task(task_id=tid)
            if t.status != "completed" or art not in t.artifacts:
                print(f"[skip] {lab}: status={t.status}")
                continue
            score = sw.read_mae(t.artifacts[art].get_local_copy())
        if a.desde:
            score = score[score.index >= pd.Timestamp(a.desde, tz="UTC")]
        hl = HL_PADRAO.get(lab, a.hl)
        h = sw.health_global(score, hl, running, t5s)
        if h.empty:
            print(f"[skip] {lab}: índice de saúde vazio")
            continue
        q = float(np.quantile(h.dropna(), a.q))
        cruz = np.array([t.timestamp() for t in h.index[h >= q]])
        # elegíveis: instantes da grade do score com a máquina rodando
        eleg = np.array([t.timestamp()
                         for t in h.index[on.reindex(h.index, method="nearest").fillna(False).values]])
        # paradas dentro do alcance do score deste braço
        alvo = [p for p in paradas if h.index[0] <= p <= h.index[-1]]
        if len(alvo) < 5 or eleg.size < len(alvo):
            print(f"[skip] {lab}: {len(alvo)} paradas no alcance do score")
            continue
        leads, plena = [], 0
        for p in alvo:
            i0 = inicio_episodio(cruz, p.timestamp(), janela_s)
            if i0 is None:
                continue
            leads.append((p.timestamp() - i0) / 3600.0)
            t_ini = pd.Timestamp(i0, unit="s", tz="UTC")
            blo = T[S].loc[p - pd.Timedelta(hours=24):p].dropna()
            val = T[S].reindex([t_ini], method="nearest")
            if len(blo) and len(val) and pd.notna(val.iloc[0]) and val.iloc[0] >= np.percentile(blo, 25):
                plena += 1
        L = np.array(leads)
        cob = len(L) / len(alvo)
        nl, nc = [], []
        for _ in range(a.perm):
            fake = rng.choice(eleg, size=len(alvo), replace=False)
            ll = [(t - inicio_episodio(cruz, t, janela_s)) / 3600.0
                  for t in fake if inicio_episodio(cruz, t, janela_s) is not None]
            if ll:
                nl.append(np.median(ll))
                nc.append(len(ll) / len(alvo))
        nl, nc = np.array(nl), np.array(nc)
        med = float(np.median(L)) if L.size else float("nan")
        r = dict(braco=lab, n_paradas=len(alvo), hl=hl, cobertura=cob,
                 cob_nulo=float(np.median(nc)), p_cobertura=float((nc >= cob).mean()),
                 lead_p50=med, lead_nulo=float(np.median(nl)),
                 p_lead=float((nl <= med).mean()), lead_p25=float(np.percentile(L, 25)),
                 frac_6h=float((L >= 6).mean()), frac_temp_plena=plena / max(len(L), 1))
        linhas.append(r)
        print(f"{lab:16s} n={len(alvo):>3}  cobertura={cob:6.1%} (nulo {r['cob_nulo']:.1%}, "
              f"p={r['p_cobertura']:.4f})  lead p50={med:5.1f}h (nulo {r['lead_nulo']:.1f}h, "
              f"p={r['p_lead']:.4f})")
        print(f"{'':16s} lead p25={r['lead_p25']:.1f}h  ≥6h antes: {r['frac_6h']:.0%}  "
              f"episódio começa com máquina em temperatura plena: {r['frac_temp_plena']:.0%}")

    if not linhas:
        raise SystemExit("nenhum braço avaliável.")
    df = pd.DataFrame(linhas)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\ngravado: {a.out}")


if __name__ == "__main__":
    main()

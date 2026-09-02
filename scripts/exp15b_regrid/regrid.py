#!/usr/bin/env python3
"""Separa NORMALIZACAO de LIMIAR no ganho do EXP15b -- a ablacao que falta no relatorio.

O problema. O relatorio compara task 14 (75,0% hit, 0,22% FP) com EXP15b (77,5%, 0,30%) e
atribui o ganho a normalizacao restrita ao periodo operacional. Mas entre os dois mudaram
DUAS coisas: a normalizacao E o limiar (THRESH_STD_K de 7,0 para 4,5; threshold absoluto
de 0,2609 para 1,0892). Baixar o limiar sozinho aumenta hit e aumenta FP -- que e
exatamente o padrao observado. Sem ablacao nao da para saber quanto de +1 alarme e
normalizacao e quanto e so sensibilidade. E a armadilha de Pareto.

O teste. Cada candidato tem uma CURVA (hit_rate x normal_alert_rate) tracada varrendo o
limiar. Se a curva do EXP15b estiver acima da do task 14 no MESMO FP, a normalizacao
melhorou o modelo. Se as curvas se sobrepuserem, o ganho e todo do limiar e some quando
os dois sao calibrados no mesmo custo.

Como e possivel offline. O limiar entra depois do modelo treinado -- os artefatos
`sequence_scores_all.csv` (MAE por sequencia) e `point_anomalies_all.csv` (estado
operacional e os dois portoes por ponto) bastam para refazer toda a cadeia sem retreinar.

VALIDACAO DA RECONSTRUCAO (feita antes de qualquer conclusao):
  - is_anom_seq a partir de mae > threshold ......... 100,0000% de concordancia
  - is_anom_point (mapeamento + portoes + escape) ... 99,99995% (2179 vs 2178 pontos)
  - hit_rate ....................................... exato nos dois (30/40 e 31/40)
  - normal_alert_rate .............................. 0,0022055 vs 0,0022040 oficial
                                                     0,0030432 vs 0,0030416 oficial
A regra de escape so fecha com o MAXIMO das 2 sequencias da janela de voto (nao o MAE da
propria sequencia) e apenas dentro do estado `on` -- detalhe que nao esta no relatorio.
"""
from __future__ import annotations
import json, pathlib
import numpy as np, pandas as pd

CACHE = pathlib.Path.home() / ".clearml/cache/storage_manager/datasets"
ALARMES = CACHE / "ds_d4c284df665e465d8492afd368837c8f/alarmes_selecionados_turbina_a.csv"
OOS = pd.Timestamp("2025-07-01")
JAN = pd.Timedelta(hours=24)
CANDS = {"task14":  "task 14 (sem norm. on-state, K=7,0)",
         "exp15r1": "EXP15 r1 (com norm., K=7,0 herdado)",
         "exp15b":  "EXP15b (com norm., K=4,5)"}


def alarmes_oos():
    a = pd.read_csv(ALARMES, low_memory=False)
    a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce")
    a = a[a.t.notna() & a["Status"].astype(str).str.startswith("ACT")]
    a = a[a["Tag Alarme"].isin(["TC382_03_A", "T5_AVG_A"]) & (a.t >= OOS)]
    return sorted(a.t)


def carrega(nome):
    s = pd.read_csv(f"{nome}_seq.csv", parse_dates=["seq_start_time"])
    p = pd.read_csv(f"{nome}_pts.csv", parse_dates=["data_datetime"]).set_index("data_datetime")
    return s, p


def avalia(s, p, ctx, thr):
    """Cadeia completa: limiar -> voto 1-de-2 -> mascara/portoes/escape -> hit e NAR."""
    an = (s["mae_TC382_03_A"] > thr).astype(int)
    votos = an.rolling(2, min_periods=2).sum().fillna(0)
    mm = s["mae_TC382_03_A"].rolling(2, min_periods=1).max()
    t_fim, val = ctx["t_fim"], ctx["val"]
    bruto = pd.Series(False, index=p.index)
    bruto.loc[t_fim[val & pd.Series((votos >= 1).values, index=t_fim.index)]] = True
    m = pd.Series(np.nan, index=p.index)
    m.loc[t_fim[val]] = mm[val].values
    fin = bruto & ctx["on"] & (~ctx["bloq"] | (m > thr * 1.5))
    hit = sum(bool(fin.loc[max(t - JAN, fin.index[0]):min(t + JAN, fin.index[-1])].any())
              for t in ctx["alarmes"])
    den = ctx["den"]
    nar = fin.values[den].sum() / max(den.sum(), 1)
    return hit, float(nar), int(fin.sum())


def main():
    alarmes = alarmes_oos()
    print(f"alarmes OOS: {len(alarmes)}\n")
    linhas = []
    for nome, rot in CANDS.items():
        cal = json.load(open(f"{nome}_cal.json"))
        ev = json.load(open(f"{nome}_eval.json"))
        s, p = carrega(nome)
        idx = p.index
        perto = np.zeros(len(idx), bool)
        for t in alarmes:
            perto |= (idx >= t - JAN) & (idx <= t + JAN)
        on = (p["operational_state"] == "on")
        ctx = dict(t_fim=s["seq_start_time"] + pd.Timedelta(seconds=59 * 30),
                   on=on, bloq=p["load_gate_blocked"] | p["volatility_gate_blocked"],
                   alarmes=alarmes,
                   den=(idx >= OOS) & ~perto & on.values)
        ctx["val"] = ctx["t_fim"].isin(idx)

        thr_of = cal["threshold"]
        h, n, np_ = avalia(s, p, ctx, thr_of)
        print(f"{rot}")
        print(f"   oficial: hit {ev['alarms_with_detected_anomaly_in_window']}/40  "
              f"NAR {ev['normal_alert_rate']:.6f}   |   reconstruido: hit {h}/40  NAR {n:.6f}", flush=True)

        grade = np.unique(np.concatenate([
            thr_of * np.geomspace(0.25, 4.0, 45), [thr_of]]))
        for thr in grade:
            h, n, np_ = avalia(s, p, ctx, thr)
            linhas.append(dict(cand=nome, rot=rot, thr=float(thr), thr_rel=float(thr / thr_of),
                               hit=h, hit_rate=h / len(alarmes), nar=n, n_pontos=np_,
                               oficial=bool(abs(thr - thr_of) < 1e-12)))
        print(f"   {len(grade)} limiares varridos", flush=True)
    T = pd.DataFrame(linhas)
    T.to_csv("regrid.csv", index=False)

    print("\n" + "=" * 92)
    print("AS TRES CURVAS, LIDAS NO MESMO FALSO ALERTA")
    print("=" * 92)
    print(f"{'NAR alvo':>10} | " + " | ".join(f"{c:>22}" for c in CANDS))
    for alvo in [0.0015, 0.0022, 0.0030, 0.0040, 0.0060, 0.0100]:
        cel = []
        for c in CANDS:
            g = T[T.cand == c].assign(d=(T[T.cand == c].nar - alvo).abs()).sort_values("d")
            r = g.iloc[0]
            cel.append(f"{int(r.hit):2d}/40 ({r.nar*100:.3f}%)")
        print(f"{alvo*100:9.2f}% | " + " | ".join(f"{x:>22}" for x in cel))
    print("\nCSV: regrid.csv")


if __name__ == "__main__":
    main()

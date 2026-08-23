#!/usr/bin/env python3
"""O quadrante que faltava: o NOSSO detector contra os alarmes de temperatura DELE.

A matriz de cobertura entre os dois detectores tinha tres celulas preenchidas e uma
vazia:

                          | alvo: alarme de temperatura | alvo: parada real
  detector EXP10c (Diego) |   92,5% (100% dos 32 reais) |   2/2 (secao 18, denominador
                          |                             |   corrigido em cruza_diego_trip)
  detector de 4 sinais    |          ???                |   8/9
                          |     <- esta celula

Sem ela nao da para dizer se a sobreposicao entre os dois e mutua ou de mao unica, nem
o que fica descoberto entre os dois.

ALVO. Os 32 alarmes fisicamente genuinos do EXP7 (secao 12.3): TC382_03_A HI (17) +
HIHI (14) + T5_AVG_A HI (1), no OOS dele (2025-07-01 a 2026-04-30). Os 8 UNDER ficam
de fora pelo motivo que ele proprio documentou e que verificamos na fonte: disparam com
valor entre -18 e -22 degC, com RUNNING_A em Comm Fail / desligado / NaN. Nao existe
precursor para prever a falha de comunicacao do proprio sensor.

REGUA. A dele, nao a nossa: janela de +-24 h em torno do alarme, e o detalhamento
preditivo (deteccao estritamente antes) / reativo (so no instante ou depois) / sem
deteccao. Ele mesmo mostrou na secao 9.5 por que o hit_rate agregado engana sem esse
detalhamento -- um candidato com FP baixo tinha 13 de 28 deteccoes preditivas, o resto
reacao tardia.

MASCARA. Reportado nas duas: a dele (maquina operando) e a nossa (operacao quente
T5>300 + blackout de 6 h pos-partida). A nossa e mais restritiva, entao pode perder
alarme que ocorre em transiente de partida -- isso precisa aparecer, nao ser escondido.

CUSTO. Convertido para a unidade dele (normal_alert_rate = fracao de pontos normais
marcados como anomalos) para ser comparavel com os 0,35% do EXP10c.

Nota de escopo: os parametros do nosso detector (k_base=1,7, k_vib=2,2) foram escolhidos
nos eventos de TREINO, anteriores a 2025-07. O OOS dele comeca exatamente em 2025-07,
entao esta avaliacao e genuinamente fora da amostra de selecao.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca

CACHE = pathlib.Path.home() / ".clearml/cache/storage_manager/datasets"
ALARMES = CACHE / "ds_d4c284df665e465d8492afd368837c8f/alarmes_selecionados_turbina_a.csv"
OOS0, OOS1 = pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2026-04-30", tz="UTC")
JAN_H = 24.0


def alvos_diego():
    a = pd.read_csv(ALARMES, low_memory=False)
    a["t"] = pd.to_datetime(a["Data da Ocorrência"], errors="coerce").dt.tz_localize("UTC")
    a = a[a["t"].notna() & a["Status"].astype(str).str.startswith("ACT")]
    a = a[(a["t"] >= OOS0) & (a["t"] <= OOS1)]
    sens = a["Tag Alarme"].isin(["TC382_03_A", "T5_AVG_A"])
    reais = a[sens & a["Condição do Alarme"].isin(["HI", "HIHI"])].sort_values("t")
    under = a[sens & (a["Condição do Alarme"] == "UNDER")].sort_values("t")
    return reais, under


def regua_diego(alerta: pd.Series, alvos: pd.DataFrame, mask: pd.Series):
    """+-24 h, com detalhamento preditivo / reativo / sem deteccao."""
    pred, reat, nada, leads, atrasos = [], [], [], [], []
    for _, r in alvos.iterrows():
        t = r["t"]
        jan = alerta.loc[t - pd.Timedelta(hours=JAN_H): t + pd.Timedelta(hours=JAN_H)]
        on = jan[jan.fillna(False)]
        if not len(on):
            nada.append(r); continue
        antes = on[on.index < t]
        if len(antes):
            pred.append(r); leads.append((t - antes.index[0]).total_seconds() / 3600)
        else:
            reat.append(r); atrasos.append((on.index[0] - t).total_seconds() / 3600)
    n = len(alvos)
    return dict(n=n, pred=len(pred), reat=len(reat), nada=len(nada),
                hit=100 * (len(pred) + len(reat)) / max(n, 1),
                taxa_pred=100 * len(pred) / max(n, 1),
                lead_med=float(np.median(leads)) if leads else np.nan,
                atraso_med=float(np.median(atrasos)) if atrasos else np.nan,
                sem_det=[r["t"].strftime("%Y-%m-%d %H:%M") for r in nada])


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask_nosso = mascara_pontuacao(df)
    mask_dele = df["in_operation"].astype(bool)          # so "operando", como no EXP7

    reais, under = alvos_diego()
    print(f"alvos no OOS dele: {len(reais)} HI/HIHI genuinos + {len(under)} UNDER descartados")
    print(f"  por tag: {reais['Tag Alarme'].value_counts().to_dict()}")
    print(f"  por condicao: {reais['Condição do Alarme'].value_counts().to_dict()}")
    dentro = ((reais["t"] >= idx[0]) & (reais["t"] <= idx[-1]))
    print(f"  dentro da cobertura do nosso cache: {int(dentro.sum())}/{len(reais)}")

    print("\nmontando 'out' ...", flush=True)
    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask_nosso, K_BASE, K_VIB)
    teto = trunca(base, 12)
    oos = (idx >= OOS0) & (idx <= OOS1)

    print("\n" + "=" * 78)
    print("O NOSSO DETECTOR NOS 32 ALARMES DE TEMPERATURA DELE (regua do EXP7, +-24 h)")
    print("=" * 78)
    print(f"{'configuracao':32s} {'hit':>7} {'pred':>6} {'reat':>6} {'nada':>6} "
          f"{'taxa pred':>10} {'lead med':>9}")
    linhas = []
    for nome, al in [("4 sinais", base), ("4 sinais + teto 12h", teto)]:
        for rot, mk in [("mascara nossa", mask_nosso), ("mascara dele (operando)", mask_dele)]:
            a2 = al if rot.startswith("mascara nossa") else (al.fillna(False) & mk)
            x = regua_diego(a2, reais, mk)
            print(f"{nome + ' | ' + rot:32s} {x['hit']:6.1f}% {x['pred']:6d} {x['reat']:6d} "
                  f"{x['nada']:6d} {x['taxa_pred']:9.1f}% {x['lead_med']:8.1f}h")
            linhas.append(dict(detector=nome, mascara=rot, **{k: v for k, v in x.items()
                                                              if k != "sem_det"}))
    print("\n  referencia EXP10c (Diego), mesmos 32 alarmes:")
    print(f"{'EXP10c':32s} {100.0:6.1f}% {29:6d} {3:6d} {0:6d} {90.6:9.1f}% {14.7:8.1f}h")

    # custo na unidade dele
    print("\n=== custo na unidade do EXP7 (normal_alert_rate) ===")
    todos = pd.concat([reais["t"], under["t"]])
    longe = pd.Series(True, index=idx)
    for t in todos:
        longe &= ~((idx >= t - pd.Timedelta(hours=JAN_H)) & (idx <= t + pd.Timedelta(hours=JAN_H)))
    for nome, al in [("4 sinais", base), ("4 sinais + teto 12h", teto)]:
        for rot, mk in [("mascara nossa", mask_nosso), ("mascara dele", mask_dele)]:
            sel = mk & oos & longe
            a2 = al.fillna(False) & mk
            print(f"  {nome + ' | ' + rot:34s} {100 * (a2 & sel).sum() / max(sel.sum(), 1):6.2f}% "
                  f"dos pontos normais  ({(a2 & sel).sum() * 2 / 60:7.1f} h em "
                  f"{sel.sum() * 2 / 60:7.0f} h)")
    print("  EXP10c (Diego)                       0.35% dos pontos normais")

    x = regua_diego(base, reais, mask_nosso)
    print(f"\n=== os {x['nada']} alarmes que o nosso detector nao ve ===")
    for s in x["sem_det"]:
        print(f"  {s}")
    pd.DataFrame(linhas).to_csv("quadrante.csv", index=False)


main()

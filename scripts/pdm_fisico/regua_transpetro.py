#!/usr/bin/env python3
"""O nosso detector medido com a regua DELES -- a do transpetro-models.

CORRECAO DE ROTA. Nas duas analises anteriores eu supus que a "regua antiga" fosse o
`predictive.py` do NOSSO repositorio (recall x FA/dia sobre span de calendario). Estava
errado. A regua que o time usa em `SIMPred-Petrobras/transpetro-models`, branch
`cabiunas/multi-failure-dates`, e outra coisa -- e a conversao que fiz antes (0,0257
FA/dia) foi construida sobre a premissa errada.

A REGUA DELES, do codigo (`src/transpetro_modelos/training/evaluate.py`,
`failure_detection_metrics_multi`):

    periodo normal  = tudo ANTES de (min(failure_dates) - normal_end_days), 60 d default
    janela pre-falha = os prefailure_days antes de CADA falha (14 d no multi, 30 d no
                       single)
    normal_alert_rate     = FRACAO DE AMOSTRAS marcadas no periodo normal
    prefailure_alert_rate = media, sobre incidentes, da fracao de amostras marcadas na
                            janela pre-falha
    composite_score       = prefailure_alert_rate * (1 - normal_alert_rate)
    discrimination_ratio  = prefailure_alert_rate / normal_alert_rate

QUATRO DIFERENCAS QUE IMPEDEM CONVERSAO DIRETA. Nao e diferenca de unidade -- e do que
esta sendo contado:
  1. eles contam AMOSTRAS, nos contamos EPISODIOS. Um alarme de 150 h e 1 episodio para
     nos e ~4.500 amostras para eles;
  2. o denominador de falso positivo deles e so o periodo ANTES da primeira falha menos
     60 d. Tudo que acontece DEPOIS da primeira falha nao entra no custo. No nosso, todo
     o tempo de operacao entra;
  3. a janela pre-falha deles e de 14 ou 30 DIAS; a nossa e de 48 h;
  4. a regua deles nao tem lead time nenhum -- "detectou" e ter marcado alguma amostra
     em 14 dias, o que nao distingue avisar com 13 dias de avisar com 10 minutos.

Este script aplica a formula DELES, tal como esta no codigo, a nossa serie de alarme.
E a unica comparacao honesta possivel sem ter a serie de alarme deles.

RESSALVA IMPORTANTE. A branch deles trabalha o **TC382_03** e reporta `n_incidents = 6`.
O nosso detector e do **TC-330.03A** com 8 eventos. Um "5/8" nao sai da configuracao
daquela branch; ou veio de outro recorte, ou de outra maquina. Sem a planilha de origem
do numero, qualquer comparacao continua sendo suposicao.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
PREFAIL_D = [14, 30]          # os dois defaults do codigo deles (multi e single)
NORMAL_END_D = 60


def alarme():
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    return pos(v, ns, REFRAT_H, DUR_MIN, False)


def metrica_deles(flags, quente, falhas, prefailure_days, normal_end_days=NORMAL_END_D):
    """Reimplementacao fiel de failure_detection_metrics_multi (evaluate.py:372).

    `flags` e a serie booleana de alarme; so amostras pontuaveis entram, que e o
    equivalente do `scores` deles (que so contem linhas pontuadas)."""
    eps = 1e-9
    s = flags[quente]
    primeira = min(pd.Timestamp(d) for d in falhas)
    normal_end = primeira - pd.Timedelta(days=normal_end_days)
    normal = s.loc[s.index < normal_end]
    normal_rate = float(normal.mean()) if len(normal) else 0.0

    por_inc = []
    for fd in falhas:
        t = pd.Timestamp(fd)
        w = s.loc[(s.index >= t - pd.Timedelta(days=prefailure_days)) & (s.index < t)]
        por_inc.append((fd, float(w.mean()) if len(w) else 0.0, len(w)))
    pre_rate = float(np.mean([r for _, r, _ in por_inc])) if por_inc else 0.0
    return dict(prefailure_alert_rate=pre_rate, normal_alert_rate=normal_rate,
                composite_score=pre_rate * (1 - normal_rate),
                discrimination_ratio=pre_rate / (normal_rate + eps),
                n_normal_samples=len(normal), por_incidente=por_inc)


if __name__ == "__main__":
    al = alarme()
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    print(f"serie pontuavel: {idx[mask][0]:%d/%m/%Y} a {idx[mask][-1]:%d/%m/%Y}")
    print(f"falhas no arquivo: {len(todas)} (a primeira em {todas.min():%d/%m/%Y})")
    print(f"nossos alvos (>= T0): {len(alvo)}\n")

    print("=" * 94)
    print("O PROBLEMA ESTRUTURAL: onde fica o PERIODO NORMAL da regua deles")
    print("=" * 94)
    for nome, fal in (("todas as falhas do arquivo", list(todas)),
                      ("so os nossos 8 alvos", list(alvo))):
        prim = min(fal); fim_normal = prim - pd.Timedelta(days=NORMAL_END_D)
        n = int((al[mask].index < fim_normal).sum())
        print(f"  {nome:<28} 1a falha {prim:%d/%m/%Y} -> normal termina "
              f"{fim_normal:%d/%m/%Y}")
        print(f"  {'':<28} amostras pontuaveis no periodo normal: {n:,} "
              f"({n*2/60:,.0f} h)")
    print("\n  -> o denominador de falso positivo deles depende inteiramente de quanto")
    print("     historico LIMPO existe antes da primeira falha. Nao mede o custo do")
    print("     detector em operacao: mede o custo dele antes da primeira falha conhecida.")

    print("\n" + "=" * 94)
    print("O NOSSO DETECTOR, MEDIDO COM A FORMULA DELES")
    print("=" * 94)
    for fal_nome, fal in (("8 alvos (TC-330.03A)", list(alvo)),
                          ("9 falhas do arquivo", list(todas))):
        for pd_ in PREFAIL_D:
            m = metrica_deles(al, mask, fal, pd_)
            print(f"\n  alvo={fal_nome}  janela pre-falha={pd_} d  "
                  f"(normal: {m['n_normal_samples']:,} amostras)")
            print(f"    prefailure_alert_rate {m['prefailure_alert_rate']:.4f}   "
                  f"normal_alert_rate {m['normal_alert_rate']:.4f}")
            print(f"    composite_score       {m['composite_score']:.4f}   "
                  f"discrimination_ratio {m['discrimination_ratio']:.1f}")
            if pd_ == 14 and fal_nome.startswith("8"):
                print(f"    por incidente (fracao da janela de {pd_} d em alarme):")
                for fd, r, n in m["por_incidente"]:
                    print(f"      {fd:%d/%m/%Y}  {r:6.3f}  ({n:,} amostras na janela)")
                acertos = sum(1 for _, r, _ in m["por_incidente"] if r > 0)
                print(f"    incidentes com ALGUM alarme na janela: {acertos}/"
                      f"{len(m['por_incidente'])}")

    print("\n" + "=" * 94)
    print("POR QUE OS NUMEROS NAO SE CONVERTEM")
    print("=" * 94)
    print("  amostra x episodio : um alarme de 150 h e 1 episodio para nos e ~4.500")
    print("                       amostras para eles. Nao ha fator de conversao fixo --")
    print("                       depende da DURACAO dos alarmes, nao da quantidade.")
    print("  denominador        : eles so contam falso positivo ANTES da 1a falha - 60 d;")
    print("                       nos contamos em todo o tempo de operacao.")
    print("  janela             : 14-30 dias contra as nossas 48 h.")
    print("  lead time          : a regua deles nao tem. 'Detectou' inclui avisar 10 min")
    print("                       antes, que operacionalmente nao serve para nada.")
    print("\n  -> '0,43' e '1,03 FP/mes' nao sao o mesmo tipo de grandeza. Para comparar,")
    print("     o unico caminho e a SERIE DE ALARME dos dois lados na mesma regua.")

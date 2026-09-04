#!/usr/bin/env python3
"""Teto de permanencia com rearme -- a alavanca de HORAS que nunca foi medida
no ponto de operacao atual.

O PROBLEMA QUE ISTO ATACA. O detector fica alarmado 1.502 h em 16 meses, 17,7%
do tempo de operacao, e o episodio mais longo dura 670 h -- 28 dias de alarme
continuo. Nenhuma regua de avaliacao pune isso (a regra C ate perdoa, porque o
episodio termina em parada real), mas e exatamente o que faz a operacao desligar
um detector: alarme que nao apaga vira ruido de fundo e perde a credibilidade que
sustenta os oito acertos.

A REGRA, como funciona numa planta. Um alarme nao fica de pe mais que TETO horas.
Passado o teto, ele e silenciado ate a condicao CAIR por pelo menos REARME horas
e subir de novo -- ai alarma outra vez, do zero. E o shelving classico de gestao
de alarme: o operador ja foi avisado; manter o alarme aceso nao acrescenta
informacao, so desgasta.

POR QUE NAO E SO TRUNCAR. Truncar sozinho perderia deteccao: o corte nao muda o
inicio do episodio, so o fim, entao um evento cujo alerta comecou 670 h antes do
trip sai da janela de 48 h. O rearme e o que devolve a deteccao -- o alerta volta
a subir perto do evento, e e essa subida que conta.

`auto_reset.py` teve a mesma ideia, mas foi medido em k=1,3/2,2 (nao no ponto de
producao), na regua antiga (antes da regra C), e hoje nem roda -- importa o pacote
`cabiunas_pdm`, apagado. Este script e autocontido.

O QUE SE MEDE, e por que estes:
  det, FP/mes, h/mes  -- a regua acordada com o time (regra C)
  horas TOTAIS em alarme e MAIOR alarme continuo -- o que a operacao sente
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo
from publica_clearml import SIN, REFRAT_H, DUR_MIN, GRID
from plota_estilo_francisco import paradas_reais_2h, classifica_regra_c

KB, KV = 1.7, 2.2
PASSO_H = pd.Timedelta(GRID).total_seconds() / 3600.0     # 2 min = 0,0333 h
TETOS = [12, 24, 48, 72, 120, None]                        # None = sem teto (atual)
REARMES = [2, 6, 12]                                       # horas de queda para rearmar


def teto_com_rearme(voto: pd.Series, teto_h, rearme_h) -> pd.Series:
    """Um alarme nao dura mais que `teto_h`; para voltar, a condicao precisa cair
    por `rearme_h` e subir de novo.

    Percorre amostra a amostra porque o estado depende do historico: quanto tempo
    este alarme ja esta de pe, e ha quanto tempo a condicao caiu."""
    if teto_h is None:
        return voto
    v = voto.to_numpy()
    out = np.zeros(len(v), dtype=bool)
    n_teto = int(round(teto_h / PASSO_H))
    n_rearme = int(round(rearme_h / PASSO_H))
    ligado = 0        # amostras que este alarme ja acumulou
    caido = n_rearme  # amostras seguidas com a condicao em baixo (comeca armado)
    silenciado = False
    for i, cond in enumerate(v):
        if cond:
            caido = 0
            if not silenciado:
                ligado += 1
                out[i] = True
                if ligado >= n_teto:      # bateu o teto: silencia ate cair
                    silenciado = True
        else:
            caido += 1
            if caido >= n_rearme:         # caiu tempo suficiente: rearma
                silenciado = False
                ligado = 0
    return pd.Series(out, index=voto.index)


def mede(al: pd.Series, paradas) -> dict:
    eps = AV.episodios(al)
    if not eps:
        return dict(det=0, fp_mes=np.nan, h_mes=np.nan, lead=np.nan,
                    h_total=0.0, maior=0.0, n_eps=0)
    m = AV.avalia(al, alvo, mask)
    meses = m["horas_op"] / 730.0
    cl = classifica_regra_c(eps, paradas)
    n_fp = sum(1 for a, b, c, l in cl if c == "FP")
    h_fp = sum((b - a).total_seconds() / 3600 for a, b, c, l in cl if c == "FP")
    duracoes = [(b - a).total_seconds() / 3600 for a, b in eps]
    return dict(det=m["det"], fp_mes=n_fp / meses, h_mes=h_fp / meses,
                lead=m["lead_med"], h_total=sum(duracoes), maior=max(duracoes),
                n_eps=len(eps))


if __name__ == "__main__":
    paradas = paradas_reais_2h()
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    voto = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])

    base = mede(pos(voto, ns, REFRAT_H, DUR_MIN, False), paradas)
    print("PONTO ATUAL (sem teto de permanencia)")
    print(f"  {base['det']}/8 · {base['fp_mes']:.3f} FP/mes · {base['h_mes']:.2f} h/mes · "
          f"lead {base['lead']:.1f}h")
    print(f"  {base['n_eps']} episodios · {base['h_total']:.0f} h TOTAIS em alarme · "
          f"maior alarme continuo {base['maior']:.0f} h\n")

    print("TETO DE PERMANENCIA COM REARME")
    print("=" * 104)
    print(f"{'teto':>6} {'rearme':>7} {'det':>5} {'FP/mes':>8} {'h/mes':>7} {'lead':>7} "
          f"{'eps':>5} {'h TOTAIS':>9} {'maior':>7}   ganho em horas totais")
    linhas = []
    for teto in TETOS:
        for rearme in ([REARMES[0]] if teto is None else REARMES):
            al = pos(teto_com_rearme(voto, teto, rearme), ns, REFRAT_H, DUR_MIN, False)
            r = mede(al, paradas)
            r.update(teto=teto, rearme=rearme)
            linhas.append(r)
            corte = 100 * (1 - r["h_total"] / base["h_total"])
            nota = ""
            if r["det"] == base["det"]:
                nota = f"-{corte:.0f}%  <- sem perder deteccao" if corte > 5 else f"-{corte:.0f}%"
            else:
                nota = f"-{corte:.0f}%  (custa {base['det'] - r['det']} deteccao)"
            t = "sem teto" if teto is None else f"{teto} h"
            print(f"{t:>6} {rearme:>6} h {r['det']:4d}/8 {r['fp_mes']:8.3f} {r['h_mes']:7.2f} "
                  f"{r['lead']:6.1f}h {r['n_eps']:5d} {r['h_total']:8.0f} h {r['maior']:6.0f} h   {nota}")

    T = pd.DataFrame(linhas)
    T.to_csv("teto_permanencia.csv", index=False)
    ok = T[(T.det == base["det"]) & T.teto.notna()]
    print("\n" + "=" * 104)
    if len(ok):
        b = ok.sort_values("h_total").iloc[0]
        print(f"MELHOR SEM CUSTO DE DETECCAO: teto de {b.teto:.0f} h, rearme de {b.rearme:.0f} h")
        print(f"  {b.det:.0f}/8 · {b.fp_mes:.3f} FP/mes · {b.h_mes:.2f} h/mes · lead {b.lead:.1f} h")
        print(f"  horas totais em alarme: {base['h_total']:.0f} -> {b.h_total:.0f} h "
              f"({100*(1-b.h_total/base['h_total']):.0f}% menos)")
        print(f"  maior alarme continuo:  {base['maior']:.0f} -> {b.maior:.0f} h")
    else:
        print("NENHUM teto preserva a deteccao -- a regra nao serve neste ponto de operacao.")
    print("-> teto_permanencia.csv")

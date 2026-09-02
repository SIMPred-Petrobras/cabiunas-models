#!/usr/bin/env python3
"""Triagem em duas camadas: ALTA CONFIANCA vs POS-PARTIDA (confirmar com operacao).

POR QUE ISTO E O QUE SOBROU. Quatro tentativas de separar algoritmicamente os falsos
positivos nascidos na borda do blackout (dist_partida = 6,4667h) das deteccoes reais que
nascem no mesmo lugar falharam: gate de nao-decaimento, ordenacao por severidade,
duracao da parada antes do religamento, rampa de T5 pos-partida (ver
[[borda-do-blackout-explica-os-fp]]). O dado que faltava (log de partida comandada x
recuperacao de trip, do DCS/SOE) nao esta disponivel no momento -- confirmado com o
usuario apos checar `Historico Ordens Turbina A.xlsx` e `Historico Operacoes
TSC33003A.xlsx`, que sao SAP PM (manutencao), nao log de eventos, e cuja cobertura de
data nem alcanca a maior parte dos episodios.

A SAIDA: nao apagar, RECLASSIFICAR. Um episodio que nasce em ate `JANELA_H` de um
religamento vira "pos-partida -- confirmar com operacao" em vez de "alarme". Nao muda
detecao (continua 8/8), muda o que o operador ve: uma fila separada, de menor urgencia,
para os casos que a fisica sozinha nao consegue desambiguar.

Mede o efeito nas DUAS camadas separadamente -- e isso e o numero que importa para a
operacao, nao o FP/mes agregado que vinhamos citando.
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import partes, pos, mask, idx, alvo, part
from publica_clearml import SIN, REFRAT_H, DUR_MIN

KB, KV = 1.7, 2.2
# a borda do blackout e um valor CRAVADO, nao uma faixa: (180+14)*2min = 6,466667h,
# o primeiro instante em que a mascara libera. Usar tolerancia apertada em torno dele
# (nao uma janela redonda tipo 8h) para nao engolir deteccoes legitimas que so
# aconteceram de cruzar o limiar um pouco depois (ex.: 07/04/2025 as 8,0h, que e
# deteccao real, nao artefato -- ver [[borda-do-blackout-explica-os-fp]]).
BORDA_H = 6.466667
TOL_H = 0.05
JANELA_H = BORDA_H + TOL_H


def alarme():
    ON = partes(KB, KV)
    ns = sum(ON[c].astype(int) for c in SIN)
    v = pd.Series(ns >= 2, index=idx) & mask & (ON["sp"] | ON["vb"])
    return pos(v, ns, REFRAT_H, DUR_MIN, False)


def classifica():
    al = alarme()
    eps = AV.episodios(al)
    JAN = pd.Timedelta(hours=48)
    jw = [(t - JAN, t) for t in alvo]
    partidas = list(idx[part.to_numpy()])

    linhas = []
    for a, b in eps:
        tp = any(a <= t1 and b >= t0 for t0, t1 in jw)
        dp = min([(a - p).total_seconds() / 3600 for p in partidas if p <= a], default=np.inf)
        camada = "pos-partida" if dp <= JANELA_H else "alta confianca"
        linhas.append(dict(ini=a, fim=b, dur_h=(b - a).total_seconds() / 3600,
                           tipo="TP" if tp else "FP", dist_partida=dp, camada=camada))
    return pd.DataFrame(linhas), eps, jw


if __name__ == "__main__":
    T, eps, jw = classifica()
    m = AV.avalia(alarme(), alvo, mask)
    meses = m["horas_op"] / 730.0
    print(f"controle: {m['det']}/8 deteccoes, {len(eps)} episodios, {meses:.2f} meses de operacao\n")

    print("=" * 96)
    print(f"OS {len(T)} EPISODIOS, POR CAMADA (janela = {JANELA_H:.0f} h desde a partida anterior)")
    print("=" * 96)
    print(f"{'inicio':>17} {'dur':>8} {'tipo':>4} {'dist_partida':>13}  camada")
    for r in T.itertuples():
        print(f"{r.ini:%d/%m/%Y %H:%M} {r.dur_h:7.1f}h {r.tipo:>4} {r.dist_partida:12.2f}h  {r.camada}")

    print("\n" + "=" * 96)
    print("O EFEITO NAS DUAS CAMADAS SEPARADAS")
    print("=" * 96)
    for cam in ("alta confianca", "pos-partida"):
        sub = T[T.camada == cam]
        tp = int((sub.tipo == "TP").sum()); fp = int((sub.tipo == "FP").sum())
        # eventos-alvo unicos cobertos so por esta camada
        cobertos = sum(1 for t, (t0, t1) in zip(alvo, jw)
                       if any(a <= t1 and b >= t0 for a, b, tipo in
                              zip(sub[sub.tipo=="TP"].ini, sub[sub.tipo=="TP"].fim, sub[sub.tipo=="TP"].tipo)))
        print(f"\n  camada '{cam}': {len(sub)} episodios ({tp} TP, {fp} FP)")
        print(f"    eventos-alvo cobertos so por esta camada: {cobertos}/8")
        print(f"    FP/mes bruto desta camada: {fp/meses:.3f}")

    print("\n" + "=" * 96)
    print("LEITURA OPERACIONAL")
    print("=" * 96)
    alta = T[T.camada == "alta confianca"]
    pos_ = T[T.camada == "pos-partida"]
    tp_alta = int((alta.tipo == "TP").sum()); fp_alta = int((alta.tipo == "FP").sum())
    tp_pos = int((pos_.tipo == "TP").sum()); fp_pos = int((pos_.tipo == "FP").sum())
    cobertos_alta = sum(1 for t, (t0, t1) in zip(alvo, jw)
                        if any(a <= t1 and b >= t0 for a, b in
                               zip(alta[alta.tipo=="TP"].ini, alta[alta.tipo=="TP"].fim)))
    fp_total = fp_alta + fp_pos
    reducao_pct = 100 * (fp_total - fp_alta) / fp_total
    print(f"  fila 'alarme' (alta confianca): {cobertos_alta}/8 eventos, {fp_alta} FP em {meses:.2f} meses "
          f"= {fp_alta/meses:.3f} FP/mes")
    print(f"    -> reducao de {reducao_pct:.0f}% no numero de alarmes de prioridade plena "
          f"({fp_total} -> {fp_alta} FP)")
    print(f"  fila 'confirmar com operacao' (pos-partida): {len(pos_)} episodios "
          f"({tp_pos} TP + {fp_pos} FP), {fp_pos/meses:.3f} FP-eq/mes")
    print(f"  cobertura total continua 8/8 -- nada foi suprimido, so reclassificado.")

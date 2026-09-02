#!/usr/bin/env python3
"""Subir o limiar reduz falso positivo? Nao. E o que custa um ponto tipo "5/8 a 0,43".

PERGUNTA. Se apertarmos o limiar para cortar falso positivo, onde chegamos? E como isso
se compara com um ponto de operacao de 5/8 a 0,43 FP/mes?

RESPOSTA CURTA, MEDIDA. O limiar nao e uma alavanca de custo neste detector. Varrendo
kb de 1,5 a 5,0 (tres vezes o ponto de operacao) e kv de 2,2 a 4,0, a deteccao desaba de
8/8 para 0/8 enquanto o custo fica PARADO na faixa de 0,86 a 1,55 FP/mes. Em kb=5,0 o
detector nao acha mais nenhuma das oito falhas e ainda assim emite 0,86 a 1,21 alarme
falso por mes.

POR QUE. Os falsos positivos nao sao excursoes pequenas encostadas no limiar -- sao
excursoes GRANDES, do mesmo tamanho ou maiores que as das falhas reais. Isso ja tinha
aparecido em severidade.py por outro caminho: o episodio de maior evidencia integrada
da serie inteira e o falso positivo de 15/10/2025, com CUSUM 43.479 contra 11.752 do
maior positivo. Subir o limiar seleciona justamente CONTRA os eventos reais, porque os
positivos de menor magnitude (26/04/2025 tem margem 1,84) morrem primeiro.

O PONTO 5/8 A 0,43. Ele existe na nossa curva -- mas nao pelo limiar. So se chega la
pelo gate geometrico de nao_decaimento.py (W=4h, delta=0,20, modo porta), e o preco esta
identificado nominalmente: perde 27/02/2025, 04/11/2025 e 26/02/2026. Este script
quantifica essa troca em vez de discuti-la, e checa quais dos eventos perdidos sao
PARADA REAL de maquina (trips_completos.csv), que e a unica classe que a operacao sente.

RESSALVA SOBRE COMPARACAO EXTERNA. Um "5/8 a 0,43" vindo de outro detector so e
comparavel depois de passar pela NOSSA regua. Ja aconteceu de a serie publicada deles
render 2/8 na nossa regua contra o 6/8 anunciado, e de os parametros da v6 nao baterem
com a configuracao documentada. Enquanto isso nao for refeito, o 5/8 a 0,43 aqui e
tratado como PONTO DE REFERENCIA sobre a nossa propria curva, nao como resultado deles.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats

MESES = 11.6
NOSSO = dict(nome="nosso ponto de operacao", det=8, fp=12, h_fp_mes=38.7, lead=29.0)
REFER = dict(nome="ponto 5/8 a 0,43", det=5, fp=5, h_fp_mes=56.9, lead=25.5)
PERDIDOS = ["2025-02-27", "2025-11-04", "2026-02-26"]


def paradas_reais():
    t = pd.read_csv("trips_completos.csv", parse_dates=["ini"])
    return set(t[t.parada_real.fillna(False).astype(bool)].ini.dt.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    print("=" * 94)
    print("1. O LIMIAR NAO E ALAVANCA DE CUSTO")
    print("=" * 94)
    d = pd.read_csv("limiar_fino.csv")
    for kv in sorted(d.kv.unique()):
        s = d[d.kv == kv]
        print(f"  kv={kv}: deteccao {s.det.iloc[0]}/8 -> {s.det.iloc[-1]}/8 "
              f"(kb {s.kb.iloc[0]} -> {s.kb.iloc[-1]}) enquanto o custo vai de "
              f"{s.fp_mes.iloc[0]:.2f} para {s.fp_mes.iloc[-1]:.2f} FP/mes")
    print(f"\n  faixa do custo em TODA a varredura: {d.fp_mes.min():.2f} a {d.fp_mes.max():.2f} FP/mes")
    print(f"  faixa da deteccao na mesma varredura: {d.det.min()}/8 a {d.det.max()}/8")
    z = d[d.det == 0]
    print(f"  com deteccao ZERO (kb >= 3,2) o custo ainda e "
          f"{z.fp_mes.min():.2f} a {z.fp_mes.max():.2f} FP/mes")
    print("  -> o custo e insensivel ao limiar; so a deteccao responde.")

    print("\n" + "=" * 94)
    print("2. O QUE O PONTO 5/8 A 0,43 DEIXA CAIR")
    print("=" * 94)
    reais = paradas_reais()
    print(f"  paradas REAIS de maquina na janela: {len(reais)} -> {sorted(reais)}")
    perd_reais = sorted(set(PERDIDOS) & reais)
    print(f"  eventos perdidos pelo ponto 5/8    : {PERDIDOS}")
    print(f"  destes, sao parada real            : {perd_reais}  "
          f"({len(perd_reais)} de {len(reais)})")
    print(f"\n  -> o ponto 5/8 mantem {len(reais)-len(perd_reais)} das {len(reais)} paradas reais.")
    print(f"     Os tres eventos que ele descarta incluem as duas unicas paradas reais")
    print(f"     de 2025-11 e 2026-02; sobra so a de 2025-12-09.")

    print("\n" + "=" * 94)
    print("3. A TROCA, EM UNIDADES QUE A OPERACAO ENTENDE")
    print("=" * 94)
    dn, dr = NOSSO, REFER
    d_falhas = dn["det"] - dr["det"]
    d_alarmes = dn["fp"] - dr["fp"]
    print(f"  {'':>24} {'falhas pegas':>13} {'alarmes falsos':>15} {'h de alarme falso':>18}")
    for x in (dn, dr):
        print(f"  {x['nome']:>24} {x['det']:>9} de 8 {x['fp']:>11} em 11,6m "
              f"{x['h_fp_mes']*MESES:>15.0f} h")
    print(f"\n  a troca: {d_alarmes} alarmes falsos a menos por {d_falhas} falhas a mais nao vistas")
    print(f"  ponto de equilibrio: so compensa se UMA falha nao vista custar menos que")
    print(f"  {d_alarmes/d_falhas:.2f} alarmes falsos.")
    print(f"\n  e o ponto 5/8 nem sequer alarma MENOS TEMPO: {dr['h_fp_mes']:.1f} h/mes contra")
    print(f"  {dn['h_fp_mes']:.1f} h/mes do nosso. Sao alarmes falsos em menor numero e MAIOR")
    print(f"  duracao -- {dr['h_fp_mes']*MESES/dr['fp']:.0f} h por alarme contra "
          f"{dn['h_fp_mes']*MESES/dn['fp']:.0f} h. A atencao gasta e maior, nao menor.")

    print("\n" + "=" * 94)
    print("4. A DIFERENCA E ESTATISTICAMENTE DISTINGUIVEL?")
    print("=" * 94)
    p_det = stats.binomtest(d_falhas, d_falhas, 0.5).pvalue / 2
    p_fp = stats.binomtest(dr["fp"], dn["fp"] + dr["fp"], 0.5).pvalue
    print(f"  deteccao 8 contra 5 nos MESMOS 8 eventos: {d_falhas} pares discordantes,")
    print(f"    todos a nosso favor -> p = {p_det:.3f} (binomial exato, unilateral)")
    print(f"  custo 12 contra 5 falsos positivos na MESMA exposicao -> p = {p_fp:.3f}")
    print(f"    (binomial, bilateral)")
    print(f"\n  Nenhum dos dois passa de 0,05 isoladamente: com 8 eventos e 11,6 meses a")
    print(f"  amostra nao resolve nem a diferenca de deteccao nem a de custo. A escolha")
    print(f"  entre os dois pontos NAO e um resultado de medicao -- e uma decisao sobre")
    print(f"  o que custa mais, uma parada nao prevista ou um alarme a atender.")

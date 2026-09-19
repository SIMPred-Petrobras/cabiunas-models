#!/usr/bin/env python3
"""A dinamica como 5o canal do v2 -- o teste que decide.

`dinamica.py` mostrou que a taxa de subida dos proprios sinais tem enriquecimento
comparavel ou melhor que o nivel, com a vantagem de nao depender de baseline
absoluto. Mas foram 36 combinacoes testadas: sob Bonferroni (alpha = 0,0014)
nenhuma sobrevive, e razoes explosivas (130x, 181x) sao sintoma de MAD degenerado
no denominador, nao de sinal.

O p-valor nao decide isso. O que decide e a metrica ponta a ponta: o canal
acrescenta deteccao na banda acionavel sem estourar o custo? Aqui os quatro
candidatos de razao MODERADA e p baixo entram no v2 pelas mesmas formas ja
testadas com a margem ao setpoint ([[margem-subsumida-pelo-v2]]), que servem de
molde -- e de aviso, porque la a complementaridade existia no sinal e era
destruida pelo pos-processamento.

RESULTADO: NAO ACRESCENTA -- e o unico "ganho" e artefato da regua.

Das 48 combinacoes (4 candidatos x 3 limiares x 4 formas), 47 empatam ou pioram.
O `Bgate` empata sempre, o que ja e conhecido: como portao, canal novo e redundante.

A excecao parecia real: `vol 3h t`, forma C, limiar em p98 ou p99 (o mesmo valor
nos dois, entao plato e nao pico) dava 0,258 FP/mes e 5,9 h/mes contra 0,344 e
6,6 -- mantendo banda 5/8, inicio 6/8, det 8/8, lead 15,7 h e os mesmos 21
episodios. Menos custo sem perder nada.

O MECANISMO DESMENTE. Olhando o que muda:

    episodio FP de 16/01 (7,4 h)     -> desaparece
    episodio novo de 17/01 a 23/01   -> 139,83 h, classificado NEUTRO

O canal troca um falso positivo de 7,4 h por um alarme de CENTO E QUARENTA HORAS
que a Regra C perdoa, por nascer perto de uma parada real. As horas caem na conta
porque as 140 h nao sao contadas -- a sala de controle veria seis dias de alarme
ininterrupto.

E A LICAO MAIOR e sobre a regua, nao sobre o canal: a Regra C existe para nao punir
o detector quando a operacao tambem viu algo, mas ela nao limita DURACAO. Um
episodio arbitrariamente longo encostado numa parada sai de graca. E a segunda vez
que uma regua nossa premia o comportamento errado -- a primeira foi a regua de
inicio creditando o 29/04 porque o detector viu MENOS
([[deploy-quebra-em-silencio]]). Qualquer ganho futuro medido em FP/mes ou h/mes
precisa passar por esta checagem: o que mudou de episodio, e quanto tempo ele dura.

Uso:  PYTHONPATH=. python dinamica_no_v2.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import cru, mask, idx, alvo, sel
from publica_clearml import (SIN, BASE, K_LO, VOTO_LO, VOTO_HI, TMIN_BANDA)
from margem_no_v2 import decide, mede, A, B, FORCA
from dinamica import slope, robusto

JAN48 = pd.Timedelta(hours=48)
POR_H = 30
m = mask.to_numpy()


def constroi(tipo: str, canal: str, horas: float) -> pd.Series:
    n = int(horas * POR_H)
    if tipo == "vol":
        v = cru[canal].rolling(n, min_periods=n // 2).std()
    elif tipo == "delta":
        s = cru[canal].ewm(halflife=pd.Timedelta("30min"), times=idx).mean()
        v = s - s.shift(n).rolling(n, min_periods=n // 2).median()
    else:
        s = cru[canal].ewm(halflife=pd.Timedelta("30min"), times=idx).mean()
        v = slope(s, horas)
    return robusto(v, m)


def voto_com(dn: pd.Series | None, forma: str, k: float) -> pd.Series:
    nA = sum(A[c].astype(int) for c in SIN)
    nB = sum(B[c].astype(int) for c in SIN)
    vA = pd.Series(nA >= VOTO_LO, index=idx) & mask
    vB = pd.Series(nB >= VOTO_HI, index=idx) & mask & (B["sp"] | B["vb"])
    if dn is None:
        return vA | vB
    d = (dn >= k).fillna(False) & mask
    if forma == "A5_4":
        vA = pd.Series((nA + d.astype(int)) >= 4, index=idx) & mask
    elif forma == "Bgate":
        vB = pd.Series(nB >= VOTO_HI, index=idx) & mask & (B["sp"] | B["vb"] | d)
    elif forma == "C":
        return vA | vB | d
    elif forma == "C2":
        return vA | vB | (d & (pd.Series(nA >= 1, index=idx) & mask))
    return vA | vB


if __name__ == "__main__":
    base = mede(decide(voto_com(None, "", 0)))
    print(f"{'':<34}{'banda':>7}{'inicio':>8}{'det':>6}{'FP/mes':>9}"
          f"{'h/mes':>8}{'lead':>8}{'eps':>6}")
    print("-" * 92)
    print(f"{'v2 publicado (controle)':<34}{base['banda']:>5}/8{base['inicio']:>6}/8"
          f"{base['det']:>4}/8{base['fp']:>9.3f}{base['h']:>8.1f}"
          f"{base['lead']:>7.1f}h{base['eps']:>6}")

    CAND = [("vol", "t", 3), ("delta", "sp", 6), ("vol", "sp", 12), ("vol", "vb", 12)]
    for tipo, c, h in CAND:
        dn = constroi(tipo, c, h)
        # limiar pelo percentil do proprio canal em regime, para duty comparavel
        print(f"\n  ---- {tipo} {h}h {c} ----")
        for pct in (95, 98, 99):
            k = float(np.nanpercentile(dn.to_numpy()[m], pct))
            duty = float(np.nanmean(dn.to_numpy()[m] >= k))
            for forma in ("A5_4", "Bgate", "C", "C2"):
                r = mede(decide(voto_com(dn, forma, k)))
                dif = []
                if r["banda"] > base["banda"]: dif.append(f"+{r['banda']-base['banda']} BANDA")
                if r["banda"] < base["banda"]: dif.append(f"{r['banda']-base['banda']} banda")
                if r["det"] < base["det"]: dif.append("perde det")
                marca = "   <<< " + ", ".join(dif) if dif else ""
                print(f"{f'p{pct} (duty {100*duty:.1f}%)  {forma}':<34}"
                      f"{r['banda']:>5}/8{r['inicio']:>6}/8{r['det']:>4}/8"
                      f"{r['fp']:>9.3f}{r['h']:>8.1f}{r['lead']:>7.1f}h{r['eps']:>6}{marca}")

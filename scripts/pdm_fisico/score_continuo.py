#!/usr/bin/env python3
"""Substituir o voto de dois niveis por uma ESTATISTICA CONTINUA.

A CRITICA AO VOTO. O gatilho atual discretiza duas vezes: cada canal vira
liga/desliga contra um limiar, e depois conta-se quantos ligaram. Um canal em
2,21 (mal cruzou) conta igual a um em 50. A magnitude, que e informacao, e
descartada na primeira etapa.

E os dois niveis sao, no fundo, duas aproximacoes da MESMA ideia:
  nivel A = "varios canais moderados"      (3 de 4, limiar baixo)
  nivel B = "poucos canais fortes"         (2 de 4, limiar alto)

Uma soma de excessos normalizados cobre os dois regimes numa expressao so:

    S = soma_c  max(0, sinal_c / limiar_c - 1)

Tres canais 20% acima do limiar dao S = 0,6; um canal 60% acima da o mesmo. O
alarme sai quando S passa de theta. Nao ha quorum, nao ha portao -- a magnitude
faz o trabalho dos dois.

VARIANTES TESTADAS, porque a forma da agregacao importa tanto quanto a ideia:
  soma      S = sum(excesso)                    -- aditiva, um canal forte basta
  soma2     S = sum(excesso^2)                  -- penaliza espalhado, premia pico
  raiz      S = sum(sqrt(excesso))              -- premia espalhado, achata pico
  quadmed   S = sqrt(mean(excesso^2))           -- media quadratica, escala fixa

O limiar de referencia de cada canal e o do NIVEL B (o alto), para que "excesso 0"
tenha o mesmo significado do gatilho especifico de hoje.

CONTROLE: o v2 publicado na mesma tabela. E a comparacao e a custo igualado --
para cada variante, theta e varrido e reporta-se o ponto de mesmo FP/mes, porque
comparar deteccao a custo diferente ja nos enganou antes
([[regrid-exp15b-ganho-e-limiar]]).

RESULTADO: REFUTADO, e o motivo explica por que o desenho atual funciona.

As 24 combinacoes (4 formas x 6 limiares) sao todas muito piores. O melhor caso
de banda e 3/8 contra 5/8 do v2, com FP de 1,550 contra 0,344 -- e conforme theta
sobe a deteccao despenca (p98 ja da 0/8 ou 1/8) enquanto o FP cai devagar.

POR QUE. A soma e DOMINADA PELO CANAL MAIS FORTE. Um canal em 50x de excesso
engole a contribuicao de tres canais moderados. Para filtrar esse canal solitario
e preciso theta alto -- e ai os eventos que dependem de varios canais moderados
nao passam. A escala continua nao consegue separar "um canal absurdo" de "tres
canais coerentes", porque ambos dao o mesmo numero.

O QUE ISSO ENSINA SOBRE O VOTO. A discretizacao nao e perda de informacao -- e o
MECANISMO DE ROBUSTEZ. Transformar cada canal em liga/desliga impede que um canal
extremo decida sozinho, e e exatamente isso que um detector com canais de duty
24-52% precisa ([[o-detector-esta-numa-fronteira]]).

E explica por que os DOIS NIVEIS existem: eles tratam "varios moderados" (A) e
"poucos fortes" (B) como REGIMES SEPARADOS, cada um com seu quorum e seu limiar.
A soma os projeta num eixo so e perde a distincao -- que e justamente a informacao
que faz o gatilho funcionar.

Uso:  PYTHONPATH=. python score_continuo.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
import avalia as AV
from pos_processamento import mask, idx, alvo, EW
from publica_clearml import SIN, BASE, TMIN_BANDA
from margem_no_v2 import decide, voto_v2
from regra_c_com_teto import classifica

JAN = pd.Timedelta(hours=48)
KH = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}

# excesso de cada canal sobre o limiar do nivel B, zerado quando abaixo
EXC = {c: (EW[c].where(mask) / (BASE[c] * KH[c]) - 1.0).clip(lower=0) for c in SIN}
E = pd.concat([EXC[c] for c in SIN], axis=1)
E.columns = SIN


def score(forma: str) -> pd.Series:
    if forma == "soma":
        return E.sum(axis=1, skipna=False)
    if forma == "soma2":
        return (E ** 2).sum(axis=1, skipna=False)
    if forma == "raiz":
        return np.sqrt(E).sum(axis=1, skipna=False)
    if forma == "quadmed":
        return np.sqrt((E ** 2).mean(axis=1, skipna=False))
    raise ValueError(forma)


def mede(fin):
    eps = AV.episodios(fin)
    det = AV.avalia(fin, alvo, mask)
    cls = classifica(eps, None)
    mes = det["horas_op"] / 730.0
    nfp = sum(1 for *_, k, _ in cls if k == "FP")
    hfp = sum(d for *_, k, d in cls if k == "FP")
    hne = sum(d for *_, k, d in cls if k == "NEUTRO")
    ban = sum(1 for t in alvo if any(t - JAN <= a <= t - pd.Timedelta(hours=TMIN_BANDA)
                                     for a, _ in eps))
    ini = sum(1 for t in alvo if any(t - JAN <= a <= t for a, _ in eps))
    lds = [(t - max([a for a, _ in eps if t - JAN <= a <= t])).total_seconds() / 3600
           for t in alvo if any(t - JAN <= a <= t for a, _ in eps)]
    return dict(banda=ban, inicio=ini, det=det["det"], nfp=nfp, fp=nfp / mes,
                h=hfp / mes, carga=(hfp + hne) / mes, eps=len(eps),
                lead=float(np.mean(lds)) if lds else float("nan"))


if __name__ == "__main__":
    b0 = mede(decide(voto_v2(None, "")))
    print(f"{'':<28}{'theta':>8}{'banda':>7}{'inicio':>8}{'det':>6}"
          f"{'FP/mes':>9}{'h/mes':>8}{'CARGA':>8}{'lead':>8}{'eps':>6}")
    print("-" * 96)
    print(f"{'v2 publicado (voto A|B)':<28}{'--':>8}{b0['banda']:>5}/8{b0['inicio']:>6}/8"
          f"{b0['det']:>4}/8{b0['fp']:>9.3f}{b0['h']:>8.1f}{b0['carga']:>8.1f}"
          f"{b0['lead']:>7.1f}h{b0['eps']:>6}")

    for forma in ("soma", "soma2", "raiz", "quadmed"):
        s = score(forma)
        v = s[mask].dropna()
        print(f"\n  ---- {forma} ----")
        # theta pelos percentis do proprio score, para varrer duty comparavel
        for pct in (90, 95, 97, 98, 99, 99.5):
            th = float(np.percentile(v, pct))
            fin = decide(pd.Series(s >= th, index=idx).fillna(False) & mask)
            r = mede(fin)
            al = []
            if r["banda"] < b0["banda"]: al.append("-banda")
            if r["det"] < b0["det"]: al.append("-det")
            if r["fp"] > b0["fp"] * 1.05: al.append("+FP")
            ok = "  <<< MELHOR" if (r["banda"] >= b0["banda"] and r["det"] >= b0["det"]
                                    and r["fp"] <= b0["fp"]) else ""
            print(f"{f'p{pct}':<28}{th:>8.2f}{r['banda']:>5}/8{r['inicio']:>6}/8"
                  f"{r['det']:>4}/8{r['fp']:>9.3f}{r['h']:>8.1f}{r['carga']:>8.1f}"
                  f"{r['lead']:>7.1f}h{r['eps']:>6}"
                  + ("  " + ",".join(al) if al else ok))

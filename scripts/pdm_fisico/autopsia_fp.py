#!/usr/bin/env python3
"""Autopsia dos falsos positivos que restam, e a hipotese dos PARES DE SINAIS.

Situacao: 23 episodios em 11,6 meses, dos quais ~21 sao falso positivo. Com esse numero da
para abrir um por um -- e diagnostico antes de prescricao e o que funcionou neste projeto
(o refratario saiu de olhar rajadas, o CUSUM de olhar deriva).

A hipotese que nunca testei: o voto >=2 trata todos os pares como equivalentes, e eles nao
sao. `t` e `p` sao erro de reconstrucao PCA sobre 14 tags de temperatura e 12 de pressao --
uma MANOBRA DE CARGA move os dois juntos por construcao, sem que nada esteja degradando.
Ja `sp` (spread do mancal) e `vb` (vibracao) medem o mesmo subsistema fisico por vias
independentes. Se os falsos positivos forem dominados por `t+p` e as deteccoes por pares
que incluem mancal, exigir pelo menos um sinal de mancal mata FP sem custar deteccao.

Isso e diferente de tudo que ja testamos: nao mexe em limiar (a curva mostrou que subir
limiar so piora), nao mexe em pos-processamento, e nao adiciona sinal. Muda o que CONTA
como confirmacao.

Reporta: cada episodio com os sinais que dispararam, a distancia ao evento mais proximo,
o estado da maquina; e a tabela de pares em FP contra pares em deteccao.
"""
from __future__ import annotations
import sys, itertools
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import BRACO
import reduz_fp as RF, cusum_cru as CC

T0 = CC.T0
JAN = pd.Timedelta(hours=48)
SIN = CC.SIN


def main():
    df = canonico(); idx = df.index
    g = pd.read_parquet("grade2min.parquet")
    todas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    sel = (idx >= T0); mask = mascara_pontuacao(df) & sel
    alvo = list(todas[todas >= T0]); m2 = mask[sel]
    op = df["in_operation"].astype(bool)
    part = op & ~op.shift(fill_value=False)
    reset = (~mask) | part
    out = roda(BRACO, df, todas)
    E = {c: out[c].ewm(halflife=pd.Timedelta(h), times=idx).mean().where(mask)
         for c, h in CC.HL.items()}
    EW = {c: DET._sustained(E[c], CC.BASE[c]*CC.K[c]) for c in SIN}
    Z = {c: (E[c] / (CC.BASE[c]*CC.K[c])).clip(upper=20) for c in SIN}
    CU = {c: pd.Series(CC.cusum_bool(Z[c], 0.75, 80, reset), index=idx) for c in SIN}
    ON = {c: (EW[c] | CU[c]) & mask for c in SIN}
    n = sum(ON[c].astype(int) for c in SIN)
    al = RF.dur_min(RF.refratario((n >= 2) & mask, 48), 60)
    eps = A.episodios(al & sel)
    jw = [(t - JAN, t) for t in alvo]
    partidas = list(idx[part & sel])
    t5 = g["T5_AVG_A"]

    print("=" * 108)
    print(f"OS {len(eps)} EPISODIOS, UM A UM")
    print("=" * 108)
    print(f"{'#':>3} {'inicio':>17} {'dur':>7} {'tipo':>10} {'sinais que dispararam':>24} "
          f"{'via':>16} {'h ate partida':>14} {'T5 med':>7}")
    linhas = []
    for i, (a, b) in enumerate(eps, 1):
        tp = "DETECCAO" if any(a <= t1 and b >= t0 for t0, t1 in jw) else "falso pos."
        w = slice(a, b)
        quais = [c for c in SIN if ON[c].loc[w].any()]
        via = []
        for c in quais:
            e_ = EW[c].loc[w].any(); u_ = CU[c].loc[w].any()
            via.append("D" if e_ and not u_ else ("C" if u_ and not e_ else "DC"))
        dp = min([(a - p).total_seconds()/3600 for p in partidas if p <= a], default=np.inf)
        dur = (b - a).total_seconds()/3600
        print(f"{i:3d} {a:%d/%m/%Y %H:%M} {dur:6.1f}h {tp:>10} {'+'.join(quais):>24} "
              f"{'/'.join(via):>16} {dp:13.1f}h {t5.loc[w].median():6.0f}C")
        linhas.append(dict(ini=a, dur=dur, tipo=tp, sinais="+".join(sorted(quais)),
                           n_sin=len(quais), dist_partida=dp, t5=float(t5.loc[w].median())))
    T = pd.DataFrame(linhas); T.to_csv("autopsia_fp.csv", index=False)

    print("\n" + "=" * 108)
    print("PARES DE SINAIS: quais confirmam deteccao, quais confirmam falso positivo")
    print("=" * 108)
    det = T[T.tipo == "DETECCAO"]; fp = T[T.tipo == "falso pos."]
    print(f"{'combinacao':>26} {'em deteccao':>12} {'em falso pos.':>14}")
    todos = sorted(set(T.sinais))
    for s in todos:
        print(f"{s:>26} {int((det.sinais==s).sum()):12d} {int((fp.sinais==s).sum()):14d}")
    print(f"\n  presenca de cada sinal:")
    for c in SIN:
        d_ = int(det.sinais.str.contains(c).sum()); f_ = int(fp.sinais.str.contains(c).sum())
        print(f"    {c:>3}: {d_}/{len(det)} das deteccoes   {f_}/{len(fp)} dos falsos positivos")
    print(f"\n  episodios com pelo menos um sinal de MANCAL (sp ou vb):")
    mm = T.sinais.str.contains("sp") | T.sinais.str.contains("vb")
    print(f"    deteccoes: {int((mm & (T.tipo=='DETECCAO')).sum())}/{len(det)}   "
          f"falsos positivos: {int((mm & (T.tipo!='DETECCAO')).sum())}/{len(fp)}")
    print(f"  episodios que sao SO processo (t e/ou p, sem mancal):")
    pp = ~mm
    print(f"    deteccoes: {int((pp & (T.tipo=='DETECCAO')).sum())}/{len(det)}   "
          f"falsos positivos: {int((pp & (T.tipo!='DETECCAO')).sum())}/{len(fp)}")

    print("\n" + "=" * 108); print("outras estruturas nos falsos positivos"); print("=" * 108)
    print(f"  distancia ate a partida anterior: mediana {fp.dist_partida.median():.1f} h   "
          f"dentro de 30 h: {int((fp.dist_partida<=30).sum())}/{len(fp)}")
    print(f"  duracao: mediana {fp.dur.median():.1f} h   "
          f"acima de 12 h: {int((fp.dur>12).sum())}/{len(fp)}")
    print(f"  numero de sinais: " +
          "  ".join(f"{k}={int((fp.n_sin==k).sum())}" for k in sorted(fp.n_sin.unique())))
    print(f"  (nas deteccoes: " +
          "  ".join(f"{k}={int((det.n_sin==k).sum())}" for k in sorted(det.n_sin.unique())) + ")")


if __name__ == "__main__":
    main()

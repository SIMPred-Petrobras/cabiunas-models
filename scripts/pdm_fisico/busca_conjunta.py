#!/usr/bin/env python3
"""Busca CONJUNTA nos sete parametros da camada de decisao, com o metodo validado.

Ate agora o ajuste foi em etapas: `k` numa rodada, o CUSUM em outra, o pos-processamento
em outra -- cada uma condicionada as anteriores. Isso encontra otimo LOCAL. Foi exatamente
esse erro que reconheci na comparacao de janelas do PCA (recalibrei 2 de 7 parametros).

Aqui os sete variam juntos: k_base, k_vib, kappa, h do CUSUM, carga residual na partida,
refratario e duracao minima. 3^7 = 2.187 configuracoes.

Criterio de selecao: MEDIA sobre a vizinhanca de grade -- validado em criterio_busca.py
(media 8/8 em todos os orcamentos; plato 8/8 em 6 de 8; proprio, min e max 6/8). Avaliacao
por LOEO ANINHADO: a busca escolhe usando 7 dos 8 eventos e o oitavo so testa.

Referencia atual: 8/8, 1,12 FP/mes, 39,0 h/mes, lead 29,0 h (media), minimo 2,8 h.
"""
from __future__ import annotations
import sys, os, itertools
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET

T0 = pd.Timestamp("2025-01-01", tz="UTC")
PAS = pd.Timedelta("2min")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
SIN = ["t", "p", "sp", "vb"]
JAN = pd.Timedelta(hours=48)
KB = [1.3, 1.7, 2.2]; KV = [1.6, 2.2, 3.0]
KA = [0.5, 0.75, 1.0]; HC = [40, 80, 160]; CR = [0.0, 0.25, 0.5]
RS = [24, 48, 72]; DS = [60, 120, 180]
EIXOS = [KB, KV, KA, HC, CR, RS, DS]
NOMES = ["kb", "kv", "ka", "h", "cr", "R", "D"]


def episodios(al, gap_h=2.0):
    v = al.fillna(False).to_numpy()
    d = np.diff(np.r_[0, v.astype(int), 0])
    br = [(al.index[a], al.index[b-1])
          for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    if not br: return []
    out = [list(br[0])]
    for s, e in br[1:]:
        if (s - out[-1][1]) <= pd.Timedelta(hours=gap_h): out[-1][1] = e
        else: out.append([s, e])
    return [tuple(x) for x in out]


def main():
    g = pd.read_parquet("grade2min.parquet"); idx = g.index
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    stable = op & (g["T5_AVG_A"] > 300)
    part = op & ~op.shift(fill_value=False)
    n_bl = int(pd.Timedelta(DET.BLACKOUT) / pd.Timedelta(C.GRID))
    sel = idx >= T0
    mask = (stable & ~part.rolling(n_bl, min_periods=1).max().astype(bool)) & sel
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    alvo = list(fal[fal >= T0]); jw = [(t - JAN, t) for t in alvo]
    alvo_s = [f"{t:%Y-%m-%d}" for t in alvo]
    meses = mask.sum()*2/60/730
    z = np.load("piso_fisico_cache.npz")
    sp = np.abs((z["b_all"] - z["med_sp"]) / z["mad_sp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((z["Xh"] - z["MED"]) / z["S"])
    vbz = np.full(len(idx), np.nan)
    vbz[z["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    vbz[~np.isfinite(vbz)] = np.nan
    out = pd.DataFrame({"t": z["t"], "p": z["p"], "sp": sp, "vb": vbz}, index=idx)
    reset = ((~mask) | part).to_numpy()
    mv = mask.values

    # EWMA por sinal (independe dos parametros)
    E = {c: out[c].ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean().where(mask)
         for c in SIN}

    def cus(x, ka, h, cr):
        xx = (x - ka).fillna(0.0).to_numpy()
        S = np.empty(len(xx)); acc = 0.0
        for i in range(len(xx)):
            acc = acc*cr if reset[i] else max(0.0, acc + xx[i]); S[i] = acc
        return S > h

    print("pre-calculando canais ...", flush=True)
    DEG, CU = {}, {}
    n = DET.SUSTAIN
    for c in SIN:
        ks = KV if c == "vb" else KB
        for k in ks:
            thr = BASE[c]*k
            DEG[(c, k)] = ((E[c] > thr).astype(int).rolling(n, min_periods=n).sum() >= n).values
            x = (E[c]/thr).clip(upper=20)
            for ka in KA:
                for h in HC:
                    for cr in CR:
                        CU[(c, k, ka, h, cr)] = cus(x, ka, h, cr)
        print(f"  {c} pronto", flush=True)

    L = []
    for kb, kv, ka, h, cr in itertools.product(KB, KV, KA, HC, CR):
        on = []
        for c in SIN:
            k = kv if c == "vb" else kb
            on.append(np.logical_or(DEG[(c, k)], CU[(c, k, ka, h, cr)]))
        voto = pd.Series((np.sum(on, axis=0) >= 2) & mv, index=idx)
        eps0 = episodios(voto)
        for R in RS:
            al = pd.Series(False, index=idx); bloq = None
            for a, b in eps0:
                if bloq is not None and a <= bloq: continue
                al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=R)
            eps1 = episodios(al)
            for D in DS:
                fin = pd.Series(False, index=idx)
                for a, b in eps1:
                    if (b - a).total_seconds()/60 + 2 >= D: fin.loc[a:b] = True
                eps = episodios(fin & sel)
                det = [f"{t:%Y-%m-%d}" for t in alvo
                       if any(a <= t and b >= t - JAN for a, b in eps)]
                fpe = [(a, b) for a, b in eps
                       if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
                hh = sum((b-a).total_seconds()/3600 + 2/60 for a, b in fpe)
                leads = []
                for t in alvo:
                    w = fin.loc[t-JAN:t-PAS]; o = w[w.fillna(False)]
                    if len(o): leads.append((t - o.index[0]).total_seconds()/3600)
                L.append(dict(kb=kb, kv=kv, ka=ka, h=h, cr=cr, R=R, D=D,
                              det=len(det), eps=len(eps), fp=len(fpe)/meses,
                              hm=hh/meses, lead=float(np.mean(leads)) if leads else np.nan,
                              lead_min=float(np.min(leads)) if leads else np.nan,
                              quais=",".join(det)))
        if len(L) % 270 == 0:
            print(f"  {len(L)}/{3**7} avaliadas", flush=True)
    T = pd.DataFrame(L); T.to_csv("busca_conjunta.csv", index=False)
    print(f"\nconfiguracoes: {len(T)}", flush=True)

    pos = {tuple(EIXOS[j].index(r[NOMES[j]]) for j in range(7)): i for i, r in T.iterrows()}
    VIZ = []
    for i, r in T.iterrows():
        c = tuple(EIXOS[j].index(r[NOMES[j]]) for j in range(7)); v = []
        for e in range(7):
            for d in (-1, 1):
                cc = list(c); cc[e] += d
                if 0 <= cc[e] < len(EIXOS[e]): v.append(pos[tuple(cc)])
        VIZ.append(np.array(v, dtype=int))
    QQ = [set(str(q).split(",")) for q in T.quais]
    D_ = np.array([[a in QQ[q] for a in alvo_s] for q in range(len(T))])
    fp = T.fp.values; lead = np.nan_to_num(T.lead.values, nan=0.0)

    print("\n" + "=" * 100)
    print("LOEO ANINHADO, criterio MEDIA da vizinhanca")
    print("=" * 100)
    print(f"{'orcamento':>16} {'LOEO':>7}   escolha tipica")
    for orc in [0.9, 1.1, 1.3, 1.6]:
        ac = 0; esc = []
        for i in range(len(alvo)):
            outros = [j for j in range(len(alvo)) if j != i]
            dtr = D_[:, outros].sum(axis=1).astype(float)
            s = np.array([np.mean(np.r_[dtr[q], dtr[VIZ[q]]]) for q in range(len(T))])
            pont = np.where(fp <= orc, s*1000.0 + lead, -np.inf)
            if not np.isfinite(pont).any(): continue
            b = int(np.argmax(pont)); esc.append(b); ac += bool(D_[b, i])
        if esc:
            r = T.iloc[pd.Series(esc).mode().iloc[0]]
            print(f"{'<= '+str(orc)+' FP/mes':>16} {str(ac)+'/8':>7}   "
                  f"kb={r.kb} kv={r.kv} ka={r.ka} h={int(r.h)} cr={r.cr} "
                  f"R={int(r.R)}h D={int(r.D)}min -> {int(r.det)}/8 {r.fp:.2f} FP/mes "
                  f"{r.hm:.1f} h/mes", flush=True)

    print("\n" + "=" * 100); print("MELHORES com 8/8, ordenados por FP"); print("=" * 100)
    S = T[T.det == 8].sort_values(["fp", "hm"])
    print(f"{'kb':>5} {'kv':>5} {'ka':>5} {'h':>5} {'cr':>5} {'R':>5} {'D':>5} "
          f"{'eps':>5} {'FP/mes':>7} {'h/mes':>7} {'lead':>6} {'min':>6}")
    for _, r in S.head(12).iterrows():
        print(f"{r.kb:5.1f} {r.kv:5.1f} {r.ka:5.2f} {int(r.h):5d} {r.cr:5.2f} {int(r.R):4d}h "
              f"{int(r.D):4d}m {int(r.eps):5d} {r.fp:7.2f} {r.hm:7.1f} {r.lead:6.1f} {r.lead_min:6.1f}")
    a = T[(T.kb == 1.7) & (T.kv == 2.2) & (T.ka == 0.75) & (T.h == 80) & (T.cr == 0.25)
          & (T.R == 48) & (T.D == 120)]
    if len(a):
        r = a.iloc[0]
        print(f"\n  ponto atual: {int(r.det)}/8  {int(r.eps)} eps  {r.fp:.2f} FP/mes  "
              f"{r.hm:.1f} h/mes  lead {r.lead:.1f} (min {r.lead_min:.1f})")
    print(f"  configuracoes com 8/8: {len(S)} de {len(T)}   menor FP: {S.fp.min():.2f}")


main()

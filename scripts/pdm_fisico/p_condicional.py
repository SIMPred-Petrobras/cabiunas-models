#!/usr/bin/env python3
"""Normalizar `p` (e `t`) por CONDICAO DE OPERACAO, nao por tempo.

Diagnostico que motiva (janeiro/2025, o mes mais caro: 164,8 h de FP, 49% de duty):
    mediana do `p` em janeiro = 1,04 (ACIMA do proprio limiar) contra 0,14 no resto -- 7,5x
    exaustao T5 em janeiro = 684 degC contra 634 no resto -- +50 degC
    os quatro termopares de mancal e o tanque de oleo sobem 2 a 5 degC, proporcionalmente
    as vibracoes praticamente nao se mexem

Ou seja: a maquina rodou em CARGA MAIS ALTA o mes inteiro. O `p` e erro de reconstrucao
PCA; operar fora da variedade que o PCA aprendeu eleva o erro sem que haja anomalia. Nao
e transiente de partida (foram 3 partidas no mes) -- e regime novo.

O `sp` nao sofre porque a mediana dos tres termopares absorve o deslocamento comum. O `p`
nao tem essa protecao.

A correcao testada aqui: comparar o `p` nao com a propria historia recente, e sim com a
historia recente NA MESMA FAIXA DE CARGA. Referencia rolante estratificada por T5 --
causal, sem refazer o PCA, e ataca a causa (mudanca de regime) em vez do sintoma.

Bracos: condicionar so `p`, so `t`, ou os dois. Varredura do numero de faixas e da janela.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET

T0 = pd.Timestamp("2025-01-01", tz="UTC")
HL = {"t": "1h", "p": "1h", "sp": "30min", "vb": "30min"}
BASE = {"t": DET.THR_FAM, "p": DET.THR_FAM, "sp": DET.THR_SPREAD, "vb": 3.0}
K = {"t": 1.7, "p": 1.7, "sp": 1.7, "vb": 2.2}
SIN = ["t", "p", "sp", "vb"]
JAN = pd.Timedelta(hours=48)
NBINS = [4, 6, 8]
HORAS_REF = [200, 400, 800]
GUARDA_H = 24


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


def z_por_faixa(s, carga, valido, nbins, horas_ref, guarda_h=GUARDA_H):
    """z robusto contra a historia recente NA MESMA FAIXA DE CARGA. Causal.

    Para cada faixa, a referencia sao as ultimas `horas_ref` de amostras DAQUELA faixa,
    pulando as `guarda_h` mais recentes -- mesma construcao do `vb`, estratificada.
    """
    v = valido.to_numpy()
    q = np.full(len(s), -1, dtype=int)
    cv = carga.where(valido).dropna()
    if len(cv) < 100:
        return pd.Series(np.nan, index=s.index)
    lim = np.quantile(cv, np.linspace(0, 1, nbins + 1))
    lim[0] -= 1e6; lim[-1] += 1e6
    q[v] = np.clip(np.digitize(carga.to_numpy()[v], lim) - 1, 0, nbins - 1)
    x = s.to_numpy().astype("float64")
    z = np.full(len(s), np.nan)
    n_ref = int(horas_ref * 30)          # amostras de 2 min
    n_g = int(guarda_h * 30)
    for b in range(nbins):
        pos = np.flatnonzero((q == b) & v & np.isfinite(x))
        if len(pos) < 200: continue
        xb = pd.Series(x[pos])
        ref = xb.shift(n_g // max(1, int(len(pos)/max(v.sum(), 1) * 1)) if False else 1)
        # janela em NUMERO DE AMOSTRAS DA FAIXA proporcional a fracao dela
        frac = len(pos) / max(v.sum(), 1)
        nb = max(60, int(n_ref * frac))
        ng = max(5, int(n_g * frac))
        refb = xb.shift(ng)
        med = refb.rolling(nb, min_periods=nb // 4).median()
        mad = (refb - med).abs().rolling(nb, min_periods=nb // 4).median() * 1.4826
        zz = ((xb - med) / mad.replace(0, np.nan)).abs()
        z[pos] = zz.to_numpy()
    return pd.Series(z, index=s.index)


def main():
    g = pd.read_parquet("grade2min.parquet"); idx = g.index
    op = (g["RUNNING_A"] > 0.5).fillna(False)
    t5 = pd.to_numeric(g["T5_AVG_A"], errors="coerce")
    stable = op & (t5 > 300)
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

    def avalia(ON):
        voto = pd.Series(sum(ON[c].astype(int) for c in SIN) >= 2, index=idx) & mask
        al = pd.Series(False, index=idx); bloq = None
        for a, b in episodios(voto):
            if bloq is not None and a <= bloq: continue
            al.loc[a:b] = True; bloq = b + pd.Timedelta(hours=48)
        fin = pd.Series(False, index=idx)
        for a, b in episodios(al):
            if (b - a).total_seconds()/60 + 2 >= 120: fin.loc[a:b] = True
        eps = episodios(fin & sel)
        det = [f"{t:%Y-%m-%d}" for t in alvo if any(a <= t and b >= t - JAN for a, b in eps)]
        fp = [(a, b) for a, b in eps if not any(a <= t1 and b >= t0 for t0, t1 in jw)]
        h = sum((b-a).total_seconds()/3600 + 2/60 for a, b in fp)
        leads = []
        for t in alvo:
            w = fin.loc[t - JAN:t - pd.Timedelta("2min")]; o = w[w.fillna(False)]
            if len(o): leads.append((t - o.index[0]).total_seconds()/3600)
        # custo de janeiro isolado
        jn = (idx >= pd.Timestamp("2025-01-01", tz="UTC")) & (idx < pd.Timestamp("2025-02-01", tz="UTC"))
        hj = sum((b-a).total_seconds()/3600 for a, b in fp
                 if a >= pd.Timestamp("2025-01-01", tz="UTC") and a < pd.Timestamp("2025-02-01", tz="UTC"))
        return dict(det=len(det), eps=len(eps), fp=len(fp)/meses, hm=h/meses,
                    lead=float(np.mean(leads)) if leads else np.nan,
                    lead_min=float(np.min(leads)) if leads else np.nan,
                    h_jan=hj, perde=",".join(sorted(set(alvo_s) - set(det))))

    def canais(serie, c, thr_mult=1.0):
        E = serie.ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean().where(mask)
        thr = BASE[c]*K[c]*thr_mult
        n = DET.SUSTAIN
        deg = ((E > thr).astype(int).rolling(n, min_periods=n).sum() >= n)
        x = (E/thr).clip(upper=20)
        xx = (x - 0.75).fillna(0.0).to_numpy()
        S = np.empty(len(xx)); acc = 0.0
        for i in range(len(xx)):
            acc = acc*0.25 if reset[i] else max(0.0, acc + xx[i]); S[i] = acc
        return (deg | pd.Series(S > 80, index=idx)) & mask

    base_ON = {c: canais(out[c], c) for c in SIN}
    r0 = avalia(base_ON)
    print(f"REFERENCIA: {r0['det']}/8  {r0['eps']} eps  {r0['fp']:.2f} FP/mes  {r0['hm']:.1f} h/mes  "
          f"lead {r0['lead']:.1f} h (min {r0['lead_min']:.1f})  janeiro {r0['h_jan']:.0f} h\n", flush=True)

    print(f"{'braco':>22} {'faixas':>7} {'ref':>6} {'det':>6} {'eps':>5} {'FP/mes':>7} {'h/mes':>7} "
          f"{'lead':>6} {'min':>6} {'jan':>7}  perde")
    L = []
    for quais in [("p",), ("t",), ("t", "p")]:
        for nb in NBINS:
            for hr in HORAS_REF:
                ON = dict(base_ON)
                for c in quais:
                    zc = z_por_faixa(out[c], t5, mask, nb, hr)
                    # o z condicional tem escala propria: limiar em unidades de sigma robusto
                    E = zc.ewm(halflife=pd.Timedelta(HL[c]), times=idx).mean().where(mask)
                    n = DET.SUSTAIN
                    deg = ((E > 3.0*K[c]).astype(int).rolling(n, min_periods=n).sum() >= n)
                    x = (E/(3.0*K[c])).clip(upper=20)
                    xx = (x - 0.75).fillna(0.0).to_numpy()
                    S = np.empty(len(xx)); acc = 0.0
                    for i in range(len(xx)):
                        acc = acc*0.25 if reset[i] else max(0.0, acc + xx[i]); S[i] = acc
                    ON[c] = (deg | pd.Series(S > 80, index=idx)) & mask
                r = avalia(ON)
                rot = "cond " + "+".join(quais)
                print(f"{rot:>22} {nb:7d} {hr:5d}h {r['det']:4d}/8 {r['eps']:5d} {r['fp']:7.2f} "
                      f"{r['hm']:7.1f} {r['lead']:6.1f} {r['lead_min']:6.1f} {r['h_jan']:6.0f}h  "
                      f"{r['perde'] if r['perde'] else '—'}", flush=True)
                L.append(dict(braco=rot, nbins=nb, ref=hr, **r))
    pd.DataFrame(L).to_csv("p_condicional.csv", index=False)


main()

#!/usr/bin/env python3
"""Piso ABSOLUTO em unidade fisica no denominador, no lugar do piso relativo (phi).

Por que este teste existe
-------------------------
piso_escala.py testou o piso relativo `s = max(s_local, phi*s_global)` e o resultado
foi ruim de um jeito especifico: o phi comprime a amplitude do duty (4,9x -> 3,6x)
mas DEIXA A TENDENCIA MAIS MONOTONA (rho 0,70 -> 1,00, p 0,233 -> 0,017) e ainda
custa 2 das 9 deteccoes. Ou seja: nao corrige a deriva, so encolhe tudo.

A hipotese do porque: o ancoramento do phi e o proprio `s_global`, calculado sobre o
historico ate o instante -- que e EXPANSIVO e cuja composicao muda junto com as
campanhas. Se o ancoramento tambem encolhe, o piso acompanha o colapso que deveria
travar. Um piso ABSOLUTO (em degC para o spread do mancal, em um para a vibracao) nao
tem essa realimentacao: ele nao sabe nada sobre o historico recente.

Duas armadilhas que este script evita de proposito
--------------------------------------------------
1. ARMADILHA DE PARETO (ja nos pegou 4x). Subir o piso baixa TODOS os z de uma vez:
   cai o custo e cai a deteccao juntos. Comparar piso a k fixo mede sensibilidade,
   nao mecanismo. piso_escala.py nao fez isso -- varreu phi com K_BASE/K_VIB fixos --
   entao o "custa 2 deteccoes" do phi pode ser em parte recuperavel baixando k.
   Aqui todo piso e comparado A CUSTO IGUALADO: para cada piso varremos k e tomamos o
   k cujo custo (h de FP por mes de operacao) mais se aproxima do custo da base.
2. GRADE ESCOLHIDA DEPOIS DE OLHAR A FISICA. A grade dos pisos e derivada dos MADs
   observados (fase 1), nao chutada -- vai de "abaixo de todo MAD" (efeito nulo,
   controle) ate "acima do maior MAD" (satura, controle do outro lado).

Fases
-----
1. pre()  : UMA passada. Walk-forward mensal para t e p (que o piso NAO altera),
            guardando o spread fisico bruto + (med, mad) de cada mes, e a referencia
            rolante da vibracao guardando (med, s) por sonda. Cacheado em .npz.
2. diag() : os denominadores em unidade fisica por semestre -- e o ancoramento do phi
            junto, para verificar se ele encolhe (a explicacao proposta da falha).
3. varre(): piso x k, a custo igualado, medindo deteccao, custo e tendencia do duty.
"""
from __future__ import annotations
import sys, itertools, os
import numpy as np, pandas as pd
from scipy import stats

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, mascara_pontuacao, ScorerMax
from ablacao4 import alerta_2k
from portoes import K_BASE, K_VIB
from auto_reset import trunca

PAS = pd.Timedelta("2min")
POR_H = 30
CACHE = "piso_fisico_cache.npz"
HORAS_BASE, GUARDA_H, PASSO_H, EXCL_D = 400.0, 24.0, 6.0, 7.0


def p_exato(y):
    n = len(y); r0 = stats.spearmanr(np.arange(n), y).statistic
    t = [abs(stats.spearmanr(np.arange(n), [y[i] for i in pm]).statistic)
         for pm in itertools.permutations(range(n))]
    return float(np.mean(np.array(t) >= abs(r0) - 1e-12)), r0


# ------------------------------------------------------------------ fase 1
def ref_rolante_bruta(X, quente, falhas):
    """rolante.z_rolante, mas devolvendo (med, s) em vez do z ja dividido.

    Guardar o denominador cru e o que permite varrer piso sem refazer as medianas
    moveis -- que sao a parte cara. Devolve tambem o s_global do phi, por passo,
    para o diagnostico de por que o piso relativo falhou.
    """
    idx = X.index
    hot = np.flatnonzero(quente.to_numpy())
    Xh = X.to_numpy()[hot].astype("float64")
    Xr = Xh.copy()
    th = idx[hot]
    for f in falhas:
        m = (th >= f - pd.Timedelta(days=EXCL_D)) & (th <= f + pd.Timedelta(days=2))
        Xr[np.asarray(m)] = np.nan

    n_base, guarda = int(HORAS_BASE * POR_H), int(GUARDA_H * POR_H)
    passo = max(1, int(PASSO_H * POR_H))
    n, K = Xh.shape
    MED = np.full((n, K), np.nan); S = np.full((n, K), np.nan); G = np.full((n, K), np.nan)
    glob_s = None
    for k in range(0, n, passo):
        fim = k - guarda; ini = max(0, fim - n_base)
        if fim - ini < n_base // 4:
            continue
        W = Xr[ini:fim]
        med = np.nanmedian(W, axis=0)
        s = np.nanmedian(np.abs(W - med), axis=0) * 1.4826
        if glob_s is None or (k // passo) % 20 == 0:
            Gw = Xr[:fim:20]
            gm = np.nanmedian(Gw, axis=0)
            glob_s = np.nanmedian(np.abs(Gw - gm), axis=0) * 1.4826
        s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
        j = min(n, k + passo)
        MED[k:j] = med; S[k:j] = s; G[k:j] = glob_s
    return hot, Xh, MED, S, G


def pre(df, falhas):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f"cache {CACHE} reaproveitado", flush=True)
        return {k: z[k] for k in z.files}

    stable = df["stable"].astype(bool)
    idx = df.index
    print("fase 1a: referencia rolante da vibracao (med, s por sonda) ...", flush=True)
    V = df[C.VIBRATION_TAGS].where(stable)
    hot, Xh, MED, S, G = ref_rolante_bruta(V, stable, falhas)

    print("fase 1b: walk-forward mensal de t, p e do spread ...", flush=True)
    meses = pd.date_range(idx[0].normalize().replace(day=1), idx[-1], freq="MS", tz="UTC")
    n = len(idx)
    t = np.full(n, np.nan); p = np.full(n, np.nan)
    b_all = DET._spread_mancal(df).to_numpy().astype("float64")   # spread fisico, degC
    med_sp = np.full(n, np.nan); mad_sp = np.full(n, np.nan)
    for i, m0 in enumerate(meses):
        m1 = meses[i + 1] if i + 1 < len(meses) else idx[-1] + PAS
        fit = df.loc[stable & (idx < m0), C.SENSOR_TAGS].dropna().tail(DET.FIT_POINTS)
        if len(fit) < DET.FIT_POINTS // 4:
            continue
        sel = (idx >= m0) & (idx < m1)
        if not sel.any():
            continue
        w = df.loc[sel]
        st = ScorerMax().fit(fit[C.TEMPERATURE_TAGS])
        sp_ = ScorerMax().fit(fit[C.PRESSURE_TAGS])
        t[sel] = st.score(w[C.TEMPERATURE_TAGS])["pca_recon"].to_numpy()
        p[sel] = sp_.score(w[C.PRESSURE_TAGS])["pca_recon"].to_numpy()
        b = DET._spread_mancal(fit)
        med_sp[sel] = float(b.median())
        mad_sp[sel] = float((b - b.median()).abs().median() * 1.4826)
        print(f"  {m0:%Y-%m}  med={med_sp[sel.argmax()]:6.2f} degC  "
              f"mad={mad_sp[sel.argmax()]:5.3f} degC", flush=True)

    d = dict(hot=hot, Xh=Xh, MED=MED, S=S, G=G, t=t, p=p,
             b_all=b_all, med_sp=med_sp, mad_sp=mad_sp)
    np.savez_compressed(CACHE, **d)
    return d


# ------------------------------------------------------------------ fase 2
def sinais(d, idx, piso_sp, piso_vb):
    """Reconstroi (t, p, sp, vb) com os pisos absolutos aplicados. Barato."""
    mad = np.maximum(d["mad_sp"], piso_sp)
    sp = np.abs((d["b_all"] - d["med_sp"]) / mad)
    S = np.maximum(d["S"], piso_vb) if piso_vb > 0 else d["S"]
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = np.abs((d["Xh"] - d["MED"]) / S)
    zmax = np.full(len(idx), np.nan)
    zmax[d["hot"]] = np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1)
    zmax[~np.isfinite(zmax)] = np.nan
    return pd.DataFrame({"t": d["t"], "p": d["p"], "sp": sp, "vb": zmax}, index=idx)


def contexto(df, falhas):
    idx = df.index
    mask = mascara_pontuacao(df)
    sems = [x.index[0] for _, x in pd.Series(idx, index=idx).groupby(pd.Grouper(freq="2QS"))
            if len(x) and (mask & (idx >= x.index[0]) & (idx <= x.index[-1])).sum() * 2 / 60 >= 300]
    jan = [((idx >= a) & (idx <= (sems[i+1] if i+1 < len(sems) else idx[-1] + PAS)))
           for i, a in enumerate(sems)]
    jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]
    return mask, sems, jan, jw


def duty_sem(al, mask, jan, jw):
    d = []
    for sel in jan:
        eps = A.episodios(al & sel)
        fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jw)]
        h = sum((b - a).total_seconds() / 3600 + 2/60 for a, b in fp)
        d.append(100 * h / max((mask & sel).sum() * 2 / 60, 1))
    return d


def diag(d, df, mask, sems, jan):
    idx = df.index
    print("\n" + "=" * 78)
    print("FASE 2 -- os denominadores em unidade fisica, por semestre")
    print("=" * 78)
    print("Se o MAD nao encolhe em degC/um, um piso absoluto nao tem o que travar e o")
    print("teste morre aqui. Se encolhe, a faixa observada define a grade do sweep.\n")
    print(f"{'semestre':>10} {'mad_sp(degC)':>13} {'med_sp(degC)':>13} "
          f"{'s_vib med(um)':>14} {'s_vib min(um)':>14} {'anc.phi(um)':>12}")
    hot_idx = idx[d["hot"]]
    for i, a in enumerate(sems):
        sel = jan[i] & mask.to_numpy()
        ms = np.nanmedian(d["mad_sp"][sel]); vs = np.nanmedian(d["med_sp"][sel])
        h = (hot_idx >= a) & (hot_idx < (sems[i+1] if i+1 < len(sems) else idx[-1] + PAS))
        sv = np.nanmedian(d["S"][h]); sv_min = np.nanmedian(np.nanmin(d["S"][h], axis=1))
        gv = np.nanmedian(d["G"][h])
        print(f"{a:%Y-%m}    {ms:13.3f} {vs:13.2f} {sv:14.3f} {sv_min:14.3f} {gv:12.3f}")
    q = np.nanpercentile(d["mad_sp"][mask.to_numpy()], [5, 25, 50, 75, 95])
    qv = np.nanpercentile(d["S"][np.isfinite(d["S"])], [5, 25, 50, 75, 95])
    print(f"\n  mad_sp  p5/p25/p50/p75/p95 (degC): " + " ".join(f"{v:.3f}" for v in q))
    print(f"  s_vib   p5/p25/p50/p75/p95 (um)  : " + " ".join(f"{v:.3f}" for v in qv))
    return q, qv


# ------------------------------------------------------------------ fase 3
def mede(out, mask, falhas, jan, jw, k_base, k_vib, teto=None):
    al = alerta_2k(out, mask, k_base, k_vib)
    if teto:
        al = trunca(al, teto)
    x = A.avalia(al, falhas, mask)
    dd = duty_sem(al, mask, jan, jw)
    pr, rr = p_exato(dd)
    return dict(det=x["det"], eps=x["episodios"], fp_mes=x["fp_mes"], h_mes=x["h_fp_mes"],
                lead=x["lead_med"], rho=rr, p_rho=pr,
                duty=",".join(f"{v:.2f}" for v in dd),
                quais=",".join(x["detectados"]), al=al, x=x)


def varre(d, df, falhas, mask, jan, jw, pisos_sp, pisos_vb, ks, teto=None):
    idx = df.index
    linhas = []
    for psp, pvb in pisos_sp:
        out = sinais(d, idx, psp, pvb)
        for kb in ks:
            r = mede(out, mask, falhas, jan, jw, kb, K_VIB, teto)
            r.pop("al"); r.pop("x")
            linhas.append(dict(piso_sp=psp, piso_vb=pvb, k_base=kb, k_vib=K_VIB, **r))
            print(f"  sp={psp:5.3f} vb={pvb:5.3f} k={kb:4.2f}  {r['det']}/9  "
                  f"{r['h_mes']:6.1f} h/mes  {r['fp_mes']:5.2f} FP/mes  "
                  f"rho={r['rho']:+.2f} p={r['p_rho']:.3f}", flush=True)
    return pd.DataFrame(linhas)


# ---------------------------------------------------- fase 3, caminho rapido
def ewmas(out, mask):
    """As 4 EWMA nao dependem de k -- calcula uma vez por piso, varre k em cima."""
    idx = out.index
    return {c: out[c].ewm(halflife=pd.Timedelta(hl), times=idx).mean().where(mask)
            for c, hl in [("t", "1h"), ("p", "1h"), ("sp", "30min"), ("vb", "30min")]}


def alerta_rapido(E, mask, k_base, k_vib):
    n = (DET._sustained(E["t"], DET.THR_FAM * k_base).astype(int)
         + DET._sustained(E["p"], DET.THR_FAM * k_base).astype(int)
         + DET._sustained(E["sp"], DET.THR_SPREAD * k_base).astype(int)
         + DET._sustained(E["vb"], 3.0 * k_vib).astype(int))
    return (n >= 2) & mask


def mede_al(al, mask, falhas, jan, jw):
    x = A.avalia(al, falhas, mask)
    dd = duty_sem(al, mask, jan, jw)
    pr, rr = p_exato(dd)
    return dict(det=x["det"], eps=x["episodios"], fp_mes=x["fp_mes"], h_mes=x["h_fp_mes"],
                lead=x["lead_med"], rho=rr, p_rho=pr,
                duty=",".join(f"{v:.2f}" for v in dd), quais=",".join(x["detectados"]))


KS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.45, 1.6, 1.7, 1.85, 2.0, 2.2, 2.5, 2.8, 3.2]
PISOS_SP = [0.0, 0.35, 0.70, 1.05, 1.60, 2.10]     # degC, equivalentes ao phi 0,1..0,6
PISOS_VB = [0.0, 0.26, 0.53, 0.80, 1.20]           # um, idem (ancora ~2,6 um)


def braco(nome, d, df, falhas, mask, jan, jw, combos):
    idx = df.index
    linhas = []
    for psp, pvb in combos:
        out = sinais(d, idx, psp, pvb)
        E = ewmas(out, mask)
        for kb in KS:
            al = alerta_rapido(E, mask, kb, K_VIB)
            r = mede_al(al, mask, falhas, jan, jw)
            rt = mede_al(trunca(al, 12), mask, falhas, jan, jw)
            linhas.append(dict(braco=nome, piso_sp=psp, piso_vb=pvb, k_base=kb, k_vib=K_VIB,
                               **r, **{f"t12_{k}": v for k, v in rt.items()}))
        print(f"  [{nome}] piso_sp={psp:.2f} degC  piso_vb={pvb:.2f} um  ({len(KS)} valores de k)",
              flush=True)
    return pd.DataFrame(linhas)


def custo_igualado(T, alvo_h, col_h="h_mes"):
    out = []
    for _, g in T.groupby(["braco", "piso_sp", "piso_vb"], sort=False):
        g = g.assign(dist=(g[col_h] - alvo_h).abs()).sort_values("dist")
        out.append(g.iloc[0])
    return pd.DataFrame(out)


def mostra(T, alvo_h, titulo, pref=""):
    M = custo_igualado(T, alvo_h, f"{pref}h_mes")
    print(f"\n--- {titulo}: custo igualado a {alvo_h:.1f} h/mes ---")
    print(f"{'piso_sp':>8} {'piso_vb':>8} {'k':>5} {'det':>6} {'h/mes':>7} {'FP/mes':>7} "
          f"{'lead':>6} {'rho':>6} {'p_rho':>6}  duty por semestre (%)")
    for _, r in M.iterrows():
        print(f"{r['piso_sp']:8.2f} {r['piso_vb']:8.2f} {r['k_base']:5.2f} "
              f"{int(r[pref+'det']):4d}/9 {r[pref+'h_mes']:7.1f} {r[pref+'fp_mes']:7.2f} "
              f"{r[pref+'lead']:6.1f} {r[pref+'rho']:+6.2f} {r[pref+'p_rho']:6.3f}  "
              f"{r[pref+'duty']}")
    return M


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    d = pre(df, falhas)
    mask, sems, jan, jw = contexto(df, falhas)
    diag(d, df, mask, sems, jan)

    out0 = sinais(d, df.index, 0.0, 0.0)
    E0 = ewmas(out0, mask)
    al0 = alerta_rapido(E0, mask, K_BASE, K_VIB)
    base = mede_al(al0, mask, falhas, jan, jw)
    base12 = mede_al(trunca(al0, 12), mask, falhas, jan, jw)
    print("\n" + "=" * 78)
    print(f"BASE piso 0, k={K_BASE}/{K_VIB}:  {base['det']}/9  {base['eps']} eps  "
          f"{base['fp_mes']:.2f} FP/mes  {base['h_mes']:.1f} h/mes  lead {base['lead']:.1f} h")
    print(f"  duty: {base['duty']}  rho={base['rho']:+.2f} p={base['p_rho']:.3f}")
    print(f"  +teto 12h: {base12['det']}/9  {base12['h_mes']:.1f} h/mes  "
          f"duty {base12['duty']}  rho={base12['rho']:+.2f} p={base12['p_rho']:.3f}")
    print("=" * 78, flush=True)

    print("\nbraco A -- piso so no spread (vibracao intacta)", flush=True)
    A_ = braco("sp", d, df, falhas, mask, jan, jw, [(x, 0.0) for x in PISOS_SP])
    print("\nbraco B -- piso so na vibracao (spread intacto)", flush=True)
    B_ = braco("vb", d, df, falhas, mask, jan, jw, [(0.0, x) for x in PISOS_VB])
    print("\nbraco C -- os dois juntos (o que o phi amarrava)", flush=True)
    C_ = braco("ambos", d, df, falhas, mask, jan, jw,
               [(s_, v_) for s_, v_ in zip(PISOS_SP[1:], PISOS_VB[1:])])
    T = pd.concat([A_, B_, C_], ignore_index=True)
    T.to_csv("piso_fisico.csv", index=False)

    for pref, alvo, tit in [("", base["h_mes"], "SEM teto"),
                            ("t12_", base12["h_mes"], "COM teto de 12 h")]:
        mostra(T, alvo, tit, pref)
    print("\nCSV: piso_fisico.csv")


if __name__ == "__main__":
    main()

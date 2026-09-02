#!/usr/bin/env python3
"""Residuo contra um modelo FISICO, no lugar de z contra a propria historia recente.

O problema que isto ataca. Os quatro sinais comparam a maquina com ELA MESMA recentemente.
Por isso o denominador pode colapsar e o custo derivar: a referencia e historia, e historia
muda. Um residuo contra um modelo fisico tem referencia FIXA -- a fisica nao deriva.

O modelo. Em regime permanente, a temperatura do metal do mancal acima da temperatura do
oleo de entrada e o calor de atrito dividido pela capacidade de remocao:

    dT = T_mancal - T_oleo  ~  Q_atrito(carga, rotacao) / (vazao * cp)

Tags disponiveis no cache, identificadas pelo catalogo de alarmes:
    TI_0305   temperatura do metal do mancal radial LNA CP  (a que dispara TAHH_6240305)
    TI_0325   temperatura do tanque de oleo lubrificante    -> entrada do oleo
    PI_0339   pressao do header de oleo                     -> proxy de vazao
    T5_AVG_A  temperatura media de exaustao                 -> proxy de carga

Residuo = dT_observado - f(carga, pressao de oleo), com f ajustada UMA VEZ em operacao
saudavel e congelada. Unidade: graus Celsius. Interpretavel pelo operador ("este mancal
esta 8 degC acima do que a fisica preve para esta carga"), o que nenhum z-score e.

ESTE SCRIPT NAO CONSTROI DETECTOR. Ele responde as tres perguntas que decidem se vale:
  1. dT depende da carga de forma limpa? (se nao, nao ha modelo a ajustar)
  2. o residuo e ESTAVEL no tempo em operacao saudavel? (se derivar, nao resolve nada e
     so troca um problema por outro)
  3. o residuo SOBE antes dos eventos? (se nao, e um modelo bonito e inutil)
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

PDM = ("/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-"
       "dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/"
       "scratchpad/pdm/src")
sys.path.insert(0, PDM)
from ablacao import canonico, mascara_pontuacao

T0 = pd.Timestamp("2024-02-01 00:00", tz="UTC")
MANCAL, OLEO, POIL, CARGA = "954005_624_TI_0305", "954005_624_TI_0325", "954005_624_PI_0339", "T5_AVG_A"
FIM_FIT = pd.Timestamp("2024-12-31", tz="UTC")     # f ajustada so em 2024, congelada depois


def main():
    df = canonico(); idx = df.index
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    mask = mascara_pontuacao(df) & (idx >= T0)

    tm = pd.to_numeric(g[MANCAL], errors="coerce")
    to = pd.to_numeric(g[OLEO], errors="coerce")
    po = pd.to_numeric(g[POIL], errors="coerce")
    ca = pd.to_numeric(g[CARGA], errors="coerce")
    # sentinelas: 871 degC e leituras negativas de termopar quebrado
    ok = (tm.between(20, 200)) & (to.between(20, 120)) & (po.between(0.5, 10)) & (ca > 300)
    m = mask & ok
    dT = (tm - to).where(m)
    print(f"pontos utilizaveis: {int(m.sum())} de {int(mask.sum())} da mascara "
          f"({100*m.sum()/max(mask.sum(),1):.1f}%)\n")

    print("=" * 92); print("1) dT depende da carga de forma limpa?"); print("=" * 92)
    print(f"  dT = TI_0305 - TI_0325 :  mediana {dT.median():.1f} degC   "
          f"p5 {dT.quantile(.05):.1f}   p95 {dT.quantile(.95):.1f}")
    q = pd.qcut(ca.where(m), 8, duplicates="drop")
    tab = pd.DataFrame({"dT": dT, "carga": ca.where(m), "poil": po.where(m), "q": q}).dropna()
    print(f"\n  {'faixa de T5 (degC)':>24} {'n':>8} {'T5 med':>8} {'dT med':>8} {'dT dp':>7} {'P oleo':>7}")
    for k, gg in tab.groupby("q", observed=True):
        print(f"  {str(k):>24} {len(gg):8d} {gg.carga.median():8.0f} {gg.dT.median():8.2f} "
              f"{gg.dT.std():7.2f} {gg.poil.median():7.2f}")
    r_c = stats.spearmanr(tab.carga, tab.dT); r_p = stats.spearmanr(tab.poil, tab.dT)
    print(f"\n  Spearman(carga, dT) = {r_c.statistic:+.3f}   Spearman(P_oleo, dT) = {r_p.statistic:+.3f}")

    print("\n" + "=" * 92)
    print("2) o residuo e estavel no tempo? (f ajustada so em 2024 e CONGELADA)")
    print("=" * 92)
    jan = pd.Series(False, index=idx)
    for t in falhas:
        jan |= (idx >= t - pd.Timedelta(days=7)) & (idx <= t + pd.Timedelta(days=2))
    fit = tab[(tab.index <= FIM_FIT) & ~jan.reindex(tab.index).fillna(False)]
    print(f"  ajuste em {len(fit)} pontos de 2024 (sem janelas de falha)")
    # f nao-parametrica: mediana de dT numa grade 2D de (carga, pressao de oleo)
    bc = np.quantile(fit.carga, np.linspace(0, 1, 13))
    bp = np.quantile(fit.poil, np.linspace(0, 1, 7))
    fit_ = fit.assign(ic=np.clip(np.digitize(fit.carga, bc) - 1, 0, len(bc) - 2),
                      ip=np.clip(np.digitize(fit.poil, bp) - 1, 0, len(bp) - 2))
    F = fit_.groupby(["ic", "ip"], observed=True).dT.median()
    glob = float(fit.dT.median())

    def prediz(carga, poil):
        ic = np.clip(np.digitize(carga, bc) - 1, 0, len(bc) - 2)
        ip = np.clip(np.digitize(poil, bp) - 1, 0, len(bp) - 2)
        return np.array([F.get((a, b), glob) for a, b in zip(ic, ip)])

    tab = tab.assign(pred=prediz(tab.carga.values, tab.poil.values))
    tab = tab.assign(res=tab.dT - tab.pred)
    fora = ~jan.reindex(tab.index).fillna(False)
    print(f"  residuo em operacao saudavel: mediana {tab.res[fora].median():+.2f} degC   "
          f"dp {tab.res[fora].std():.2f} degC")
    print(f"\n  {'semestre':>10} {'n':>8} {'residuo mediano':>16} {'dp':>7} | "
          f"{'para comparar: dT bruto':>24}")
    for k, gg in tab[fora].groupby(pd.Grouper(freq="2QS"), observed=True):
        if len(gg) < 500: continue
        print(f"  {k:%Y-%m} {len(gg):8d} {gg.res.median():15.2f} degC {gg.res.std():7.2f} | "
              f"{gg.dT.median():23.2f}")
    sem = [(k, gg.res.median()) for k, gg in tab[fora].groupby(pd.Grouper(freq="2QS"), observed=True) if len(gg) >= 500]
    if len(sem) >= 3:
        rr = stats.spearmanr(np.arange(len(sem)), [v for _, v in sem])
        print(f"\n  tendencia do residuo por semestre: rho={rr.statistic:+.2f} p={rr.pvalue:.3f} "
              f"(n={len(sem)})   amplitude {max(v for _,v in sem)-min(v for _,v in sem):.2f} degC")

    print("\n" + "=" * 92)
    print("3) o residuo sobe antes dos eventos?")
    print("=" * 92)
    base = tab.res[fora]
    p95, p99 = base.quantile(.95), base.quantile(.99)
    print(f"  referencia saudavel: p95 = {p95:+.2f} degC   p99 = {p99:+.2f} degC\n")
    print(f"  {'evento':>17} {'modo':>9} {'res max 48h':>12} {'res med 48h':>12} "
          f"{'h acima do p99':>15} {'pontos':>7}")
    fal = pd.read_csv("falhas.csv", parse_dates=["evento"])
    fal["evento"] = fal["evento"].dt.tz_convert("UTC")
    for _, r in fal.iterrows():
        t = r["evento"]
        if t < T0: continue
        w = tab.res.loc[t - pd.Timedelta(hours=48):t]
        mo = "mancal" if "Manc" in str(r["alarmes"]) else ("oleo" if "Óleo" in str(r["alarmes"]) else "selagem")
        if not len(w):
            print(f"  {t:%d/%m/%Y %H:%M} {mo:>9} {'sem dado':>12}"); continue
        print(f"  {t:%d/%m/%Y %H:%M} {mo:>9} {w.max():+11.2f} {w.median():+11.2f} "
              f"{int((w > p99).sum())*2/60:14.1f} h {len(w):7d}")


if __name__ == "__main__":
    main()

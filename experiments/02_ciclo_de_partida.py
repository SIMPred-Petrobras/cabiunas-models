#!/usr/bin/env python3
"""EXP 02 — ATLAS, PARTE 2: O CICLO DE PARTIDA (regimes, risco por tempo desde a partida, acomodação das tags).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR. Exploratório.

H3 DO EXP 01 JÁ ESTÁ REFUTADA e não se testa: dos três eventos sem corrida terminal distintiva, só 04/11/2025 aconteceu perto de uma partida (15,3 h); 17/03/2025 foi a 211 h e 26/02/2026 a 676 h da última partida.

PERGUNTAS.
  P1  EXPOSIÇÃO E RISCO. Quantas horas de máquina ligada existem por faixa de tempo desde a partida (tau)? Qual é a taxa de trips por 1.000 h em cada faixa, com IC exato de Poisson, e a concentração nas primeiras 32 h é maior que o acaso sob risco uniforme?
  P2  TIPO DE PARTIDA. As 159 partidas se dividem por duração da parada anterior (breve 10-30 min, 30 min-12 h, 12-72 h, > 72 h). O risco (trip em 48 h) depende do tipo? (O M25 da pesquisa viu 3 de 36 depois de paradas breves reais, p = 0,14, pós-hoc.)
  P3  ACOMODAÇÃO. Em quantas horas depois de uma partida cada tag se acomoda (passa a ficar dentro de +-3 sigma do seu estado permanente por >= 24 h seguidas)? Isso diz se o blackout fixo de 6 h do v2 é adequado e quais tags exigem mais.

DEFINIÇÕES (fixadas agora).
  · ligada suave: RUNNING_A > 0,5 com lacunas menores que 10 min preenchidas (oscilação de tag não é parada). Partida = subida da ligada suave; a parada anterior é o trecho desligado imediatamente antes (>= 10 min por construção). tau = horas desde a última partida.
  · Período: 2024-02-01 (início da janela oficial de eventos) até o último dado real (21/04/2026 00:56). Eventos: os 8 trips-alvo (a partir de 27/02/2025); o trip de 16/01/2024 fica fora do período.
  · Exposição = horas ligadas por faixa de tau: [0,1), [1,3), [3,6), [6,12), [12,24), [24,48), [48,96), [96,200), [200, inf).
  · Teste de concentração: o nº de trips com tau <= 32 h contra Binomial(8, fração da exposição com tau <= 32 h), unilateral.
  · Acomodação: estado permanente de cada tag = mediana e MAD robusto (piso de 5% da mediana de |x|) das amostras ligadas com tau > 72 h; por corrida (trecho ligado contínuo) com >= 30 h, z horário = mediana horária de |x - mediana| / MAD; tau_acomoda = a primeira hora a partir da qual TODAS as horas seguintes (até 24 h, ou até o fim da corrida) têm z <= 3.

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  · A fração da exposição ligada com tau <= 32 h fica entre 15% e 35%, e o p do teste de concentração (5 de 8 a <= 32 h) é < 0,05 (~65%).
  · A taxa de trips nas primeiras 32 h é 3 a 6 vezes a taxa depois (IC largo, 5 eventos contra 3).
  · O tipo da partida NÃO tem efeito estatisticamente claro (Fisher p > 0,05 em qualquer corte), por falta de eventos (~80%).
  · Pelo menos 8 das 35 tags têm mediana de tau_acomoda >= 6 h (o blackout de 6 h é curto para elas); as sondas de vibração são as mais lentas (mediana de tau_acomoda >= 6 h em >= 5 das 10); a temperatura T5 acomoda em <= 3 h.
  · Há >= 60 corridas com >= 30 h.

LIMITES. 8 eventos; faixas de tau escolhidas aqui; o estado permanente usa a própria história (inclui a deriva); 5 dos 8 trips a <= 31 h é o número de partida da pesquisa e aqui se recalcula com outra definição de partida (ligada suave).

Uso:  PYTHONPATH=src .venv/bin/python experiments/02_ciclo_de_partida.py
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro

SAIDA = Path(__file__).resolve().parents[1] / "data_local" / "atlas"
FAIXAS = [0, 1, 3, 6, 12, 24, 48, 96, 200, np.inf]
H = 1 / 30                                          # horas por amostra de 2 min


def ligada_suave(g):
    lig = (g["RUNNING_A"] > 0.5).fillna(False).to_numpy()
    out = lig.copy(); n = len(lig); i = 0
    while i < n:
        if not lig[i]:
            j = i
            while j < n and not lig[j]:
                j += 1
            if 0 < i and j < n and (j - i) < 5:          # lacuna de < 10 min entre dois trechos ligados
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    g = nvio.carrega_grade()
    fim = g.index[g["RUNNING_A"].notna() & g[[c for c in g.columns if c.startswith("TV_")]].notna().any(axis=1)][-1]
    g = g.loc[pd.Timestamp("2024-02-01", tz="UTC"):fim]; idx = g.index; lig = ligada_suave(g); n = len(g)
    sub = np.flatnonzero(lig[1:] & ~lig[:-1]) + 1                             # subidas
    start = np.full(n, -1); last = -1; k = 0
    for i in range(n):
        if k < len(sub) and i == sub[k]:
            last = i; k += 1
        start[i] = last
    tau = np.where(lig & (start >= 0), (np.arange(n) - start) * H, np.nan)   # horas desde a última partida (só enquanto ligada)
    # duração da parada anterior a cada partida
    off_prev = []
    for i in sub:
        j = i - 1
        while j >= 0 and not lig[j]:
            j -= 1
        off_prev.append((i - j - 1) * H if j >= 0 else np.nan)
    off_prev = np.array(off_prev)
    print(f"período {idx[0]:%Y-%m-%d} a {idx[-1]:%Y-%m-%d %H:%M}: {int(lig.sum() * H)} h ligadas | {len(sub)} partidas (ligada suave)", flush=True)

    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    tau_T = []
    for T in reg:
        i = idx.searchsorted(T) - 1                                           # a última amostra ligada antes de T
        tau_T.append(float(tau[i]) if i >= 0 and np.isfinite(tau[i]) else np.nan)
    print("tau (h desde a última partida) nos 8 trips:", [round(x, 1) for x in tau_T])

    print("\nP1. EXPOSIÇÃO E RISCO POR FAIXA DE tau")
    linhas = []
    for a, b in zip(FAIXAS[:-1], FAIXAS[1:]):
        expo = float(np.sum((tau >= a) & (tau < b))) * H
        nt = sum(1 for x in tau_T if np.isfinite(x) and a <= x < b)
        lo, hi = (stats.chi2.ppf(0.025, 2 * nt) / 2 if nt else 0.0, stats.chi2.ppf(0.975, 2 * nt + 2) / 2)
        linhas.append(dict(faixa_h=f"[{a:g},{b:g})", exposicao_h=round(expo), trips=nt, taxa_1000h=round(1000 * nt / expo, 2) if expo > 0 else np.nan,
                           ic95=f"[{1000 * lo / expo:.2f}; {1000 * hi / expo:.2f}]" if expo > 0 else ""))
    T1 = pd.DataFrame(linhas); print(T1.to_string(index=False))
    expo_tot = float(np.sum(np.isfinite(tau))) * H; expo_32 = float(np.sum(tau <= 32)) * H
    k32 = sum(1 for x in tau_T if np.isfinite(x) and x <= 32); pc = expo_32 / expo_tot
    p = stats.binomtest(k32, len(reg), pc, alternative="greater").pvalue
    r_in, r_out = k32 / expo_32, (len(reg) - k32) / (expo_tot - expo_32)
    print(f"   fração da exposição ligada com tau <= 32 h: {100 * pc:.1f}% ({expo_32:.0f} de {expo_tot:.0f} h) | trips a <= 32 h: {k32} de {len(reg)} | p binomial unilateral = {p:.4f} | taxa dentro/fora = {r_in / r_out:.1f}x")

    print("\nP2. TIPO DE PARTIDA (parada anterior) E TRIP EM 48 H")
    cortes = [(0, 0.5, "breve (10-30 min)"), (0.5, 12, "30 min-12 h"), (12, 72, "12-72 h"), (72, 1e9, "> 72 h")]
    tipo = np.full(len(sub), -1)
    for q, (a, b, _) in enumerate(cortes):
        tipo[(off_prev >= a) & (off_prev < b)] = q
    trip_48 = np.array([any(0 <= (T - idx[i]).total_seconds() / 3600 <= 48 for T in reg) for i in sub])
    L = []
    for q, (a, b, nome) in enumerate(cortes):
        m = tipo == q; L.append(dict(tipo=nome, partidas=int(m.sum()), com_trip_48h=int(trip_48[m].sum()), taxa=round(float(trip_48[m].mean()), 3) if m.any() else np.nan))
    print(pd.DataFrame(L).to_string(index=False)); print(f"   no total: {int(trip_48.sum())} de {len(sub)} partidas seguidas de trip em 48 h")
    for q in range(len(cortes)):
        m = tipo == q
        if m.sum() and trip_48[m].sum():
            f = stats.fisher_exact([[int(trip_48[m].sum()), int(m.sum() - trip_48[m].sum())], [int(trip_48[~m].sum()), int((~m).sum() - trip_48[~m].sum())]], alternative="greater")
            print(f"   Fisher unilateral ({cortes[q][2]} contra as demais): p = {f[1]:.3f}, razão de chances {f[0]:.1f}")
    print("   parada anterior à última partida de cada trip (h):", [round(float(off_prev[np.flatnonzero(sub <= idx.searchsorted(T))[-1]]), 1) if (sub <= idx.searchsorted(T)).any() else None for T in reg])

    print("\nP3. ACOMODAÇÃO DAS TAGS DEPOIS DA PARTIDA (horas até ficar dentro de +-3 sigma do estado permanente por >= 24 h)")
    tags = [c for c in g.columns if c not in ("RUNNING_A", "HSX_6240001A")]; X = g[tags].to_numpy("float64")       # T5_AVG_A entra: a acomodação dela interessa (a máscara usa T5 > 300)
    perm = np.isfinite(tau) & (tau > 72)
    med = np.nanmedian(X[perm], axis=0); mad = 1.4826 * np.nanmedian(np.abs(X[perm] - med), axis=0)
    sig = np.maximum(np.maximum(mad, 0.05 * np.nanmedian(np.abs(X[perm]), axis=0)), 1e-6)
    cortes_run = np.flatnonzero(np.diff(np.concatenate(([0], lig.astype(int), [0]))) != 0).reshape(-1, 2)
    corridas = [(a, b) for a, b in cortes_run if (b - a) * H >= 30 and a in set(sub)]
    print(f"   corridas (partida até a parada) com >= 30 h: {len(corridas)} de {len(sub)}")
    res = {t: [] for t in tags}
    for a, b in corridas:
        d = pd.DataFrame(np.abs(X[a:b] - med) / sig, columns=tags, index=idx[a:b]); zh = d.groupby(np.arange(b - a) // 30).median().to_numpy()
        for j, t in enumerate(tags):
            z = zh[:, j]; ok = np.where(np.isfinite(z), z <= 3.0, False); ts = np.nan
            for k0 in range(len(z)):
                w = ok[k0:k0 + 24]
                if len(w) >= 6 and w.all() and np.isfinite(z[k0:k0 + 24]).all():
                    ts = float(k0); break
            res[t].append(ts)
    S = pd.DataFrame({t.replace('954005_624_', ''): dict(corridas=len(v), nunca_acomoda=f"{100 * np.mean(np.isnan(v)):.0f}%", mediana_h=round(float(np.nanmedian(v)), 1) if np.isfinite(v).any() else np.nan,
                                                         p90_h=round(float(np.nanquantile(v, 0.9)), 1) if np.isfinite(v).any() else np.nan) for t, v in res.items()}).T
    print(S.sort_values("mediana_h", ascending=False).to_string())
    lentas = [t for t in tags if np.isfinite(res[t]).any() and np.nanmedian(res[t]) >= 6]
    vib_len = [t for t in lentas if t.startswith("TV_")]
    print(f"\n   tags com mediana de acomodação >= 6 h: {len(lentas)} de {len(tags)} | sondas de vibração entre elas: {len(vib_len)} de 10 | T5_AVG_A acomoda em (mediana) {np.nanmedian(res['T5_AVG_A']):.1f} h")

    fig, ax = plt.subplots(figsize=(8, 3.2)); ax.bar(range(len(T1)), T1["taxa_1000h"].fillna(0)); ax.set_xticks(range(len(T1))); ax.set_xticklabels(T1["faixa_h"], rotation=30, fontsize=7)
    ax.set_ylabel("trips por 1.000 h ligadas"); ax.set_title("risco por tempo desde a partida (8 trips)"); fig.tight_layout(); fig.savefig(SAIDA / "risco_por_tau.png", dpi=90); plt.close(fig)
    S.to_csv(SAIDA / "acomodacao_por_tag.csv")


if __name__ == "__main__":
    main()

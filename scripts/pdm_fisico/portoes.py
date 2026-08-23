#!/usr/bin/env python3
"""Porta os dois portoes de supressao do EXP10b/EXP10c (Diego) para o detector fisico.

Motivo. O relatorio EXP6-EXP10c corta o falso alerta de 1,94% para 0,35% sem perder
deteccao, usando dois mecanismos de pos-processamento que nunca testamos aqui:

  rampa       (EXP10b) suprime quando a TAXA de variacao suavizada do proxy de carga
              passa de um limiar. Nao e o mesmo que exigir mais sustentacao: testamos
              sustain maior em dois_regimes.py e falhou. Aqui o gatilho e dT/dt, nao
              persistencia. Diego mediu que a janela longa (halflife 120min/janela
              360min) custa 6 preditivos porque rampa de falha e rampa de carga tem a
              mesma assinatura numa janela longa -- so a janela curta (15/30min) separa.

  volatilidade(EXP10c) desvio-padrao movel causal de 60 min de cada sonda de vibracao,
              media entre sondas, bloqueia quando o NIVEL passa do limiar. Inverte o
              que testamos em regra_vib.py: la a vibracao era requisito para alarmar
              (refutado); aqui ela e supressor. Nao depende de confiar no nivel
              absoluto da vibracao, so na sua variabilidade -- compativel com a
              quarentena declarada.

Protocolo. Os dois portoes so suprimem, entao andam sobre uma curva deteccao x FP.
Comparar "com portao" contra "sem portao" no MESMO limiar e a armadilha de Pareto em
que ja caimos duas vezes: o ganho pode ser so dessensibilizacao. Por isso este script
tambem varre k_base sem portao e compara a FP IGUALADO. O parametro do portao e
escolhido no TREINO (< 2025-07-01) e so depois se olha o teste -- e exatamente o
ponto onde o protocolo do Diego escorrega (ramp_max e o limiar de volatilidade dele
foram escolhidos para preservar os preditivos do proprio OOS).
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO

K_BASE, K_VIB = 1.7, 2.2          # ponto de operacao escolhido no treino (ablacao4)
PAS = pd.Timedelta("2min")

RAMP_MAX = [40, 60, 80, 100, 150, 200, 300, 500]        # degC/h
VOL_THR  = [0.15, 0.20, 0.25, 0.30, 0.39, 0.50, 0.70, 1.00, 1.50]   # micrometro
KS_REF   = [1.7, 1.9, 2.1, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0]


def proxy_rampa(g: pd.DataFrame, hl="15min", jan="30min") -> pd.Series:
    """Taxa de variacao suavizada do proxy de carga, em degC/h. Parametros do
    EXP10b (janela curta: e a que separa rampa de carga de rampa de falha)."""
    t5 = g["T5_AVG_A"].astype("float64")
    s = t5.ewm(halflife=pd.Timedelta(hl), times=g.index).mean()
    n = int(pd.Timedelta(jan) / PAS)
    return ((s - s.shift(n)) / (pd.Timedelta(jan) / pd.Timedelta("1h"))).abs()


def indice_volatilidade(g: pd.DataFrame, jan_min=60) -> pd.Series:
    """Desvio-padrao movel causal por sonda, reduzido pela media entre sondas.
    Mesma construcao do EXP10c, nas mesmas unidades (micrometro)."""
    V = g[C.VIBRATION_TAGS].astype("float64")
    n = int(jan_min / 2)
    return V.rolling(n, min_periods=n // 2).std().mean(axis=1)


def metricas(al, mask, falhas, tr, te):
    d = {}
    for tag, m, ev in [("tr", tr, falhas[falhas < CORTE]), ("te", te, falhas[falhas >= CORTE]),
                       ("tot", pd.Series(True, index=al.index), falhas)]:
        am, qm = al[m], (mask & m)[m]
        x = A.avalia(am, ev, qm)
        d.update({f"{tag}_det": x["det"], f"{tag}_n": x["n_ev"], f"{tag}_fp": x["fp_mes"],
                  f"{tag}_h": x["h_fp_mes"], f"{tag}_lead": x["lead_med"],
                  f"{tag}_quais": ",".join(x["detectados"])})
    return d


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    tr = pd.Series(idx < CORTE, index=idx); te = ~tr
    mask = mascara_pontuacao(df)

    print("montando 'out' (walk-forward mensal, uma vez) ...", flush=True)
    out = roda(BRACO, df, falhas)
    base = alerta_2k(out, mask, K_BASE, K_VIB)

    rate = proxy_rampa(g).reindex(idx)
    vol = indice_volatilidade(g).reindex(idx)
    q = mask & tr
    print(f"\nproxy de rampa |dT5/dt| em operacao pontuavel (treino): "
          f"p50={rate[q].median():.1f}  p95={rate[q].quantile(.95):.1f}  "
          f"p99={rate[q].quantile(.99):.1f}  max={rate[q].max():.0f} degC/h")
    print(f"indice de volatilidade da vibracao (treino): "
          f"p50={vol[q].median():.3f}  p95={vol[q].quantile(.95):.3f}  "
          f"p99={vol[q].quantile(.99):.3f}  max={vol[q].max():.2f} um\n", flush=True)

    linhas = []
    linhas.append(dict(arm="base", par=np.nan, **metricas(base, mask, falhas, tr, te)))
    for r in RAMP_MAX:
        al = base & ~(rate > r).fillna(False)
        linhas.append(dict(arm="rampa", par=r, **metricas(al, mask, falhas, tr, te)))
    for v in VOL_THR:
        al = base & ~(vol > v).fillna(False)
        linhas.append(dict(arm="volat", par=v, **metricas(al, mask, falhas, tr, te)))
    # os dois juntos, no ponto mais permissivo de cada um que nao custa deteccao no treino
    for r in RAMP_MAX:
        for v in VOL_THR:
            al = base & ~((rate > r) | (vol > v)).fillna(False)
            linhas.append(dict(arm="ambos", par=f"{r}/{v}", **metricas(al, mask, falhas, tr, te)))
    # curva de referencia: dessensibilizar sem portao nenhum
    for k in KS_REF:
        al = alerta_2k(out, mask, k, K_VIB)
        linhas.append(dict(arm="k_ref", par=k, **metricas(al, mask, falhas, tr, te)))

    t = pd.DataFrame(linhas)
    t.to_csv("portoes.csv", index=False)

    def mostra(sub, titulo):
        print(f"--- {titulo}")
        print(f"{'par':>10} {'treino':>8} {'FP/mes':>7} {'h/mes':>7} | "
              f"{'teste':>8} {'FP/mes':>7} {'h/mes':>7} | {'total':>8} {'h/mes':>7}")
        for _, r in sub.iterrows():
            print(f"{str(r['par']):>10} {r.tr_det:>4.0f}/{r.tr_n:<3.0f} {r.tr_fp:7.2f} {r.tr_h:7.1f} | "
                  f"{r.te_det:>4.0f}/{r.te_n:<3.0f} {r.te_fp:7.2f} {r.te_h:7.1f} | "
                  f"{r.tot_det:>4.0f}/{r.tot_n:<3.0f} {r.tot_h:7.1f}")
        print()

    for a, tit in [("base", "SEM PORTAO (ponto de operacao atual)"),
                   ("rampa", "PORTAO DE RAMPA -- suprime se |dT5/dt| > par (degC/h)"),
                   ("volat", "PORTAO DE VOLATILIDADE -- suprime se std(vib,60min) > par (um)"),
                   ("k_ref", "REFERENCIA: dessensibilizar k_base, sem portao")]:
        mostra(t[t.arm == a], tit)

    print("=== comparacao a FP IGUALADO (horas de alarme por mes, serie toda) ===")
    ref = t[t.arm == "k_ref"].sort_values("tot_h")
    for a in ["rampa", "volat", "ambos"]:
        s = t[t.arm == a]
        melhor = None
        for _, r in s.iterrows():
            j = (ref.tot_h - r.tot_h).abs().idxmin()
            rr = ref.loc[j]
            ganho = r.tot_det - rr.tot_det
            if melhor is None or (ganho, -r.tot_h) > melhor[0]:
                melhor = ((ganho, -r.tot_h), r, rr)
        (gan, _), r, rr = melhor
        print(f"{a:>7} par={str(r['par']):<10} {r.tot_det:.0f}/{r.tot_n:.0f} a {r.tot_h:.1f} h/mes"
              f"   vs   k={rr['par']} {rr.tot_det:.0f}/{rr.tot_n:.0f} a {rr.tot_h:.1f} h/mes"
              f"   -> {gan:+.0f} deteccao a FP igual")


main()

#!/usr/bin/env python3
"""Valida os pontos de 8/9 que o stack do EXP7 produziu contra o nosso alvo.

diego_stack.py achou dois pontos que parecem bater o detector daqui com folga:
entrada 'diego' (12 sensores) e entrada 'nosso' (36), ambas com fit estatico,
percentil 99,5 e sustentacao de 2 min -> 8/9 eventos a ~5 h de alarme/mes, contra
7/9 a 29,8 h/mes do detector de 4 sinais com teto de 12 h. Antes de acreditar, cinco
coisas -- as tres primeiras podem derrubar o resultado inteiro.

1. DENOMINADOR CONTAMINADO. O escore so existe onde TODAS as features existem: 96,7%
   do tempo na entrada 'diego', mas so 73,8% na entrada 'nosso'. Onde o escore e NaN o
   detector nao pode alarmar -- e essas horas continuam no denominador de FP/mes. Isso
   deflaciona o custo de graca. Aqui a regua e refeita sobre mask AND escore-existe, e
   a deteccao e recontada so nos eventos cuja janela de 48 h tem cobertura de escore.

2. SUSTENTACAO DE 2 MIN E DETECCAO POR PISCADA. Com sustentacao de 2 min, um unico
   ponto isolado dentro de 48 h ja conta como deteccao, enquanto o detector daqui exige
   30 min sustentados. Mede-se quanto tempo o alerta fica de fato ativo dentro da janela
   de cada evento -- se for uma piscada de 2 min, nao e alerta operacional.

3. PERMUTACAO. Com 15,7 episodios/mes espalhados, a cobertura de 48 h pode ficar alta o
   bastante para acertar 8/9 por acaso. Sem esse p o numero nao significa nada.

4. LOEO. Percentil e sustentacao escolhidos nos outros 8, testados no que ficou de fora.

5. DERIVA. O custo por semestre, que e onde o detector daqui falha (duty de 1,7% a 8,1%).
   Previ no turno anterior que um fit unico deriva MAIS que um refit; o braco 'walk' deu
   pior, o que contraria a previsao -- mede-se aqui semestre a semestre.

Alem disso: os dois detectores erram eventos DIFERENTES (o daqui perde 2024-01-16, o
stack perde 2025-03-17). Testa-se a uniao.
"""
from __future__ import annotations
import sys, glob
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
import avalia as A
from ablacao import canonico, roda, mascara_pontuacao, CORTE
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from auto_reset import trunca
from diego_stack import alerta_de, PERCENTIS, SUSTENTA_MIN

ORC = 80.0


def seleciona(s, mask, tr, ev_tr, ref_idx):
    """Escolhe percentil/sustentacao SO no treino, orcamento de horas fixado a priori."""
    ref = s[ref_idx].dropna()
    melhor = None
    for p in PERCENTIS:
        lim = float(np.percentile(ref, p))
        for sm in SUSTENTA_MIN:
            al = alerta_de(s, mask, lim, sm)
            x = A.avalia(al[tr], ev_tr, (mask & tr)[tr])
            if x["h_fp_mes"] > ORC:
                continue
            chave = (x["det"], -x["h_fp_mes"])
            if melhor is None or chave > melhor[0]:
                melhor = (chave, p, sm, lim)
    return melhor


def main():
    df = canonico()
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    mask = mascara_pontuacao(df)
    tr = pd.Series(idx < CORTE, index=idx)
    ev_tr = falhas[falhas < CORTE]

    print("montando o detector de 4 sinais (referencia) ...", flush=True)
    out = roda(BRACO, df, falhas)
    nosso = alerta_2k(out, mask, K_BASE, K_VIB)
    nosso_teto = trunca(nosso, 12)

    guardado = {}
    for f in sorted(glob.glob("escore_*_iforest_*.parquet")):
        tag = f[len("escore_"):-len(".parquet")]
        s = pd.read_parquet(f)["escore"].reindex(idx)
        tem = s.notna()
        m_just = mask & tem                      # denominador honesto (1)
        cob = 100 * (m_just.sum() / mask.sum())
        print(f"\n{'='*76}\n{tag}  --  escore existe em {cob:.1f}% do tempo pontuavel")

        sel = seleciona(s, m_just, tr, ev_tr, m_just & tr)
        if sel is None:
            print("  nenhum ponto respeita o orcamento de treino"); continue
        (_, _), p, sm, lim = sel
        al = alerta_de(s, m_just, lim, sm)
        guardado[tag] = al

        for rot, m_aval in [("denominador do grid (mask)", mask),
                            ("denominador honesto (mask AND escore)", m_just)]:
            x = A.avalia(al, falhas, m_aval)
            x.update(A.permuta(al, m_aval, x["det"], len(falhas)))
            print(f"  {rot:42s} {x['det']}/9  {x['episodios']:4d} eps  "
                  f"{x['fp_mes']:5.2f} FP/mes  {x['h_fp_mes']:6.1f} h/mes  p={x['p']:.4f}")
        print(f"  ponto: percentil={p} sustentacao={sm} min")

        # (2) o alerta e piscada ou permanencia?
        print("  tempo de alerta ATIVO dentro da janela de 48 h de cada evento:")
        for ev in falhas:
            jan = al.loc[ev - pd.Timedelta(hours=48):ev]
            hp = m_just.loc[ev - pd.Timedelta(hours=48):ev].sum() * 2 / 60
            print(f"    {ev.strftime('%Y-%m-%d')}: {jan.sum()*2/60:6.2f} h ativo "
                  f"(de {hp:5.1f} h com escore)")

        # (4) LOEO
        pega = []
        for ev in falhas:
            outros = falhas[falhas != ev]
            s2 = seleciona(s, m_just, tr, outros[outros < CORTE], m_just & tr)
            if s2 is None:
                pega.append(False); continue
            (_, _), p2, sm2, lim2 = s2
            a2 = alerta_de(s, m_just, lim2, sm2)
            pega.append(A.avalia(a2, pd.Series([ev]), m_just)["det"] == 1)
        print(f"  LOEO: {sum(pega)}/9")

        # (5) deriva por semestre
        jw = [(t - pd.Timedelta(hours=48), t) for t in falhas]
        print(f"  {'semestre':>10} {'h escore':>9} {'eps FP':>7} {'FP/mes':>7} {'h/mes':>7} {'duty%':>7}")
        for _, gser in pd.Series(idx, index=idx).groupby(pd.Grouper(freq="2QS")):
            if len(gser) == 0: continue
            se = (idx >= gser.index[0]) & (idx <= gser.index[-1])
            ho = (m_just & se).sum() * 2 / 60
            if ho < 300: continue
            eps = A.episodios(al & se)
            fp = [(a, b) for a, b in eps if not any((a <= t1) and (b >= t0) for t0, t1 in jw)]
            hfp = sum((b - a).total_seconds() / 3600 + 2/60 for a, b in fp)
            mes = ho / 730
            print(f"  {str(gser.index[0].date()):>10} {ho:9.0f} {len(fp):7d} {len(fp)/mes:7.2f} "
                  f"{hfp/mes:7.1f} {100*hfp/mes/730:7.2f}")

    # ---- uniao com o detector daqui
    print(f"\n{'='*76}\nENSEMBLE: uniao com o detector de 4 sinais")
    x = A.avalia(nosso, falhas, mask)
    xt = A.avalia(nosso_teto, falhas, mask)
    print(f"  {'4 sinais':38s} {x['det']}/9  {x['episodios']:4d} eps  {x['h_fp_mes']:6.1f} h/mes")
    print(f"  {'4 sinais + teto 12h':38s} {xt['det']}/9  {xt['episodios']:4d} eps  {xt['h_fp_mes']:6.1f} h/mes")
    for tag, al in guardado.items():
        u = (al.fillna(False) | nosso_teto.fillna(False)) & mask
        xu = A.avalia(u, falhas, mask)
        xu.update(A.permuta(u, mask, xu["det"], len(falhas)))
        xs = A.avalia(al, falhas, mask)
        print(f"  {'uniao com ' + tag:38s} {xu['det']}/9  {xu['episodios']:4d} eps  "
              f"{xu['h_fp_mes']:6.1f} h/mes  p={xu['p']:.4f}")
        print(f"     stack sozinho pega: {','.join(xs['detectados'])}")
        print(f"     4 sinais+teto pega: {','.join(xt['detectados'])}")


main()

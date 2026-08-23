#!/usr/bin/env python3
"""Tentativas de derrubar o 9/9 do crest factor. Tres ataques.

O resultado de crest.py e forte demais para aceitar: LOEO 9/9, controle de
sensibilidade passa (o detector de 4 sinais nao chega a 9/9 em NENHUMA sensibilidade
entre 90 e 199 h/mes), e o parametro e chato (k_tx de 1,0 a 4,0 dao o mesmo 9/9).
Resultado chato e bom sinal, mas nada disso testa as tres coisas abaixo.

ATAQUE 1 -- VAZAMENTO NA REFERENCIA. rolante.z_rolante recebe a lista de falhas e
  apaga +-7 dias em volta de CADA uma da serie que vira referencia. Isso e correto em
  producao (a degradacao conhecida nao deve virar o normal), mas no LOEO e vazamento:
  a referencia do fold ja sabe onde esta o evento que foi escondido, e apagar a janela
  pre-falha DEIXA A REFERENCIA MAIS LIMPA, o que INFLA o z justamente durante a falha.
  E a direcao que favorece o resultado. Aqui o z e recalculado por fold com apenas os
  8 eventos visiveis.

ATAQUE 2 -- HISTORICO CURTO EM 2024-01. O evento que o crest ganha e 2024-01-16, o
  primeiro da serie. z_rolante precisa de 400 h de operacao quente anterior e aceita
  rodar com 1/4 disso. Se o z de 16/01/2024 vier de uma referencia de 100 h mal
  cobertas, ele nao e comparavel aos demais e o ganho e artefato de borda.

ATAQUE 3 -- ALERTA DE SORTE. A k_tx=4,0 o crest acrescenta so 4 episodios e 1,3 h/mes.
  Se a deteccao de 2024-01-16 for um unico blip de 30 min, e sorte, nao sinal. Mede-se
  quanto tempo o crest fica de fato acima do limiar dentro da janela de 48 h.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C, detector as DET
import avalia as A, rolante as RO
from ablacao import canonico, roda, mascara_pontuacao
from ablacao4 import alerta_2k, BRACO
from portoes import K_BASE, K_VIB
from textura import textura, alerta_n, BASE_H, _mm

K_TX = [1.0, 1.4, 1.7, 2.1, 2.5, 3.0, 4.0]


def crest_bruto(g, stable):
    """Crest factor por sonda, antes do z rolante (nao depende de falhas)."""
    n = 30
    V = g[C.VIBRATION_TAGS].astype("float64").interpolate(limit=5)
    D = V - V.rolling(int(BASE_H * 30), min_periods=int(BASE_H * 30) // 4).median()
    F = D.abs().rolling(n, min_periods=n // 2).max() / (_mm(D ** 2, n) ** 0.5).replace(0, np.nan)
    return F.where(stable, axis=0)


def main():
    df = canonico()
    g = pd.read_parquet("grade2min.parquet")
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")
    idx = df.index
    stable = df["stable"].astype(bool)
    mask = mascara_pontuacao(df)

    print("montando 'out' e crest bruto ...", flush=True)
    out = roda(BRACO, df, falhas)
    F = crest_bruto(g, stable)

    # ---------------- ATAQUE 2: quanto historico quente existe antes de cada evento
    print("\n=== ATAQUE 2: horas de operacao quente ANTES de cada evento ===")
    print(f"serie comeca em {idx[0]}; z_rolante quer 400 h e aceita >= 100 h\n")
    hq = stable.astype(int)
    for ev in falhas:
        ate = hq.loc[:ev - pd.Timedelta(hours=24)].sum() * 2 / 60
        print(f"  {ev.strftime('%Y-%m-%d')}: {ate:8.0f} h quentes acumuladas antes "
              f"{'  <-- abaixo de 400 h' if ate < 400 else ''}")

    # ---------------- ATAQUE 1: LOEO sem vazamento (z recalculado por fold)
    print("\n=== ATAQUE 1: LOEO com o z do crest recalculado sem o evento escondido ===")
    print(f"{'evento fora':>12} {'k_tx*':>6} {'8 restantes':>12} {'h/mes':>7} | "
          f"{'detectado?':>11} {'lead':>6}")
    res = []
    for ev in falhas:
        outros = falhas[falhas != ev]
        Z = RO.z_rolante(F, stable, outros, horas_base=400, guarda_h=24, phi=0.0).max(axis=1)
        melhor = None
        for k in K_TX:
            al = alerta_n(out, mask, [(Z, "30min", 3.0)], K_BASE, K_VIB, k)
            x = A.avalia(al, outros, mask)
            chave = (x["det"], -x["h_fp_mes"])
            if melhor is None or chave > melhor[0]:
                melhor = (chave, k, x, al)
        (_, _), k, x8, al = melhor
        xe = A.avalia(al, pd.Series([ev]), mask)
        pega = xe["det"] == 1
        print(f"{ev.strftime('%Y-%m-%d'):>12} {k:6.1f} {x8['det']:>8}/8   {x8['h_fp_mes']:7.1f} | "
              f"{'SIM' if pega else 'nao':>11} {xe['lead_med'] if pega else float('nan'):6.1f}")
        res.append(dict(evento=ev, k_tx=k, det8=x8["det"], h_mes=x8["h_fp_mes"],
                        detectado=pega, lead=xe["lead_med"] if pega else np.nan))
    r = pd.DataFrame(res); r.to_csv("crest_loeo_sem_vazamento.csv", index=False)
    print(f"\nLOEO SEM vazamento: {int(r.detectado.sum())}/9   (com vazamento era 9/9)")

    # ---------------- ATAQUE 3: o alerta de 2024-01-16 e sustentado ou blip?
    print("\n=== ATAQUE 3: quanto o crest fica acima do limiar nas 48 h de cada evento ===")
    Z = RO.z_rolante(F, stable, falhas, horas_base=400, guarda_h=24, phi=0.0).max(axis=1)
    zew = Z.ewm(halflife=pd.Timedelta("30min"), times=idx).mean().where(mask)
    for k in (2.5, 4.0):
        thr = 3.0 * k
        print(f"\n  k_tx={k} (limiar z={thr:.0f}):")
        for ev in falhas:
            jan = zew.loc[ev - pd.Timedelta(hours=48):ev]
            acima = (jan > thr).fillna(False)
            print(f"    {ev.strftime('%Y-%m-%d')}: {acima.sum()*2/60:5.1f} h acima "
                  f"(de {jan.notna().sum()*2/60:5.1f} h pontuaveis)  z max={jan.max():7.1f}")


main()

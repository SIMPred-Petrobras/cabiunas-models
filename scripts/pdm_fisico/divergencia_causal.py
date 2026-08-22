#!/usr/bin/env python3
"""Mesma pergunta do divergencia_autopsia.py, mas sem o vazamento: a rodada
anterior calibrou mediana/MAD com a serie INTEIRA (passado e futuro) pra
todos os 9 eventos, nao so pro 2024-01-16 -- so aconteceu de eu checar a
versao causal so daquele. Aqui todo evento e recalibrado usando SO dado
estavel anterior a ele (o que um detector real teria disponivel), igual ao
walk-forward usado no resto do projeto.

Sem percentil-contra-fundo (o fundo tambem precisaria ser causal, por evento,
complicando sem necessidade): aqui o z ja e a unidade certa (MADs), entao
compara-se direto contra o limiar Z=5 sustentado que divergencia_termopares.py
ja usa pra chamar de episodio real.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

PDM = "/tmp/claude-1000/-home-thallys-Documents-projeto-petrobras-Analise-exploratoria-dos-dados-analise-cabiunas-cabv2-cabiunas-models/e6d62cc6-1642-437f-9af7-77c9e87ef823/scratchpad/pdm/src"
sys.path.insert(0, PDM)
from cabiunas_pdm import config as C
from ablacao import canonico, CORTE

SUAVIZA_N = 30    # 1h a 2min
Z_EPISODIO = 5.0  # mesmo limiar do script original, validado em temperatura

GRUPOS = {
    "temperatura_array": ["TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A",
                          "TC382_05_A", "TC382_06_A"],
    "vibracao": list(C.VIBRATION_TAGS),
}


def z_causal(df, tags, stable, causal_mask):
    X = df[tags]
    zs = {}
    for t in tags:
        irm = [c for c in tags if c != t]
        res = X[t] - X[irm].median(axis=1)
        base = res[causal_mask]
        m = base.median(); mad = (base - m).abs().median() * 1.4826
        mad = mad if (mad and mad > 0) else np.nan
        z = ((res - m) / mad).where(stable).rolling(SUAVIZA_N, min_periods=SUAVIZA_N // 2).median()
        zs[t] = z
    return pd.DataFrame(zs)


def main():
    df = canonico()
    stable = df["stable"].astype(bool)
    idx = df.index
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")

    linhas = []
    for nome, tags in GRUPOS.items():
        print(f"\n=== {nome} ({len(tags)} sensores) ===", flush=True)
        for ev in falhas:
            causal = stable & (idx < ev)
            n_h = causal.sum() * 2 / 60.0
            Z = z_causal(df, tags, stable, causal)
            sinal = Z.abs().max(axis=1)
            janela = sinal[(idx >= ev - pd.Timedelta(hours=24)) & (idx < ev)].dropna()
            zmax = float(janela.max()) if len(janela) else np.nan
            zmed = float(janela.mean()) if len(janela) else np.nan
            sustentado = bool((janela >= Z_EPISODIO).sum() * 2 >= 6 * 60) if len(janela) else False
            tag_conj = "teste" if ev >= CORTE else "treino"
            linhas.append(dict(grupo=nome, evento=ev, conjunto=tag_conj, horas_calibracao=n_h,
                                z_max_24h=zmax, z_medio_24h=zmed, episodio_z5_6h=sustentado))
            print(f"  {ev.strftime('%Y-%m-%d')} ({tag_conj}, {n_h:6.0f}h de calibracao): "
                  f"z_max={zmax:5.2f}  z_medio={zmed:5.2f}  "
                  f"{'>>> EPISODIO (Z>=5, 6h)' if sustentado else ''}", flush=True)

    R = pd.DataFrame(linhas)
    R.to_csv("divergencia_causal.csv", index=False)
    print("\n--- tabela final: z_max nas 24h antes de cada evento, calibracao SO com o passado ---")
    with pd.option_context("display.width", 160):
        print(R.pivot_table(index=["evento", "conjunto"], columns="grupo", values="z_max_24h")
              .to_string(float_format=lambda v: f"{v:5.2f}"))
    print(f"\neventos com episodio real (Z>=5 sustentado 6h) em algum grupo: "
          f"{R.groupby('evento')['episodio_z5_6h'].any().sum()} de {len(falhas)}")


main()

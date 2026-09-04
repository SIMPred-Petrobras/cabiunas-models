#!/usr/bin/env python3
"""Autopsia: divergencia entre irmaos antecede algum dos 9 trips?

Generaliza o metodo ja validado de divergencia_termopares.py (que so olha o
array TC382, sem nenhum modelo, e ja achou um episodio real sem alarme) pra
vibracao -- os 10 sondas X/Y sao a mesma grandeza medida em 5 posicoes, exigem
o mesmo tipo de residuo contra irmaos. Oleo fica de fora: PI e PDI medem
grandezas diferentes (pressao absoluta vs diferencial), residuo-contra-mediana
nao tem leitura fisica ali.

Metodo, identico em espirito ao divergencia_termopares.py:
  1. residuo_i = sensor_i - mediana(irmaos), so operacao quente-estavel.
  2. z robusto: centra pela MEDIANA do proprio residuo, escala pelo MAD --
     remove o offset de posicao de cada sensor.
  3. suaviza com mediana movel de 1h.
  4. sinal do grupo = max(|z_i|) sobre os irmaos -- preserva o caso de UM
     sensor divergindo sozinho (autopsia anterior achou disso).
  5. autopsia sem limiar arbitrario: percentil do sinal nas 24h antes de cada
     evento contra o fundo (resto do periodo quente-estavel, com guarda de
     72h ao redor de qualquer um dos 9 eventos) -- mesma regua do teste de
     FFT, pra ficar comparavel.

Nenhum parametro aqui foi ajustado olhando os 9 eventos.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

# O pacote `cabiunas_pdm` vive agora em ./cabiunas_pdm, restaurado da branch
# do Francisco (ver cabiunas_pdm/__init__.py). O caminho antigo era um
# diretorio temporario que foi apagado; nao ha mais sys.path a inserir.
from cabiunas_pdm import config as C
from ablacao import canonico, CORTE

SUAVIZA_N = 30      # 30 * 2min = 1h
GUARD = pd.Timedelta(hours=72)

GRUPOS = {
    "temperatura_array": ["TC382_01_A", "TC382_02_A", "TC382_03_A", "TC382_04_A",
                          "TC382_05_A", "TC382_06_A"],
    "vibracao": list(C.VIBRATION_TAGS),
}


def divergencia_grupo(df: pd.DataFrame, tags: list[str], stable: pd.Series) -> pd.Series:
    X = df[tags]
    zs = {}
    for t in tags:
        irm = [c for c in tags if c != t]
        med_irm = X[irm].median(axis=1)
        res = (X[t] - med_irm).where(stable)
        m = res.median(); mad = (res - m).abs().median() * 1.4826
        mad = mad if (mad and mad > 0) else np.nan
        z = ((res - m) / mad).rolling(SUAVIZA_N, min_periods=SUAVIZA_N // 2).median()
        zs[t] = z
    Z = pd.DataFrame(zs)
    return Z.abs().max(axis=1)


def main():
    df = canonico()
    stable = df["stable"].astype(bool)
    falhas = pd.read_csv("falhas.csv", parse_dates=["evento"])["evento"].dt.tz_convert("UTC")

    resumo = []
    for nome, tags in GRUPOS.items():
        print(f"\n=== {nome} ({len(tags)} sensores) ===", flush=True)
        sinal = divergencia_grupo(df, tags, stable)

        fundo_mask = pd.Series(True, index=sinal.index)
        for ev in falhas:
            fundo_mask &= ~((sinal.index >= ev - GUARD) & (sinal.index <= ev + GUARD))
        fundo_mask &= stable
        fundo = sinal[fundo_mask].dropna()
        print(f"  quadros de fundo: {len(fundo)}")

        for ev in falhas:
            janela = sinal[(sinal.index >= ev - pd.Timedelta(hours=24)) & (sinal.index < ev)].dropna()
            pct = 100.0 * (fundo.to_numpy() < janela.mean()).mean() if len(janela) else np.nan
            tag_conj = "teste" if ev >= CORTE else "treino"
            resumo.append(dict(grupo=nome, evento=ev, conjunto=tag_conj,
                                n_quadros_24h=len(janela), percentil=pct))
            print(f"  {ev.strftime('%Y-%m-%d')} ({tag_conj}): percentil nas 24h antes = "
                  f"{pct:5.1f}" if not np.isnan(pct) else
                  f"  {ev.strftime('%Y-%m-%d')} ({tag_conj}): sem dado suficiente")

    R = pd.DataFrame(resumo)
    R.to_csv("divergencia_autopsia.csv", index=False)
    print("\n--- tabela final (percentil do sinal de divergencia nas 24h antes do evento) ---")
    with pd.option_context("display.width", 160):
        print(R.pivot_table(index=["evento", "conjunto"], columns="grupo", values="percentil")
              .to_string(float_format=lambda v: f"{v:5.1f}"))
    print(f"\nmedia geral dos percentis: {R['percentil'].mean():.1f}  (52 no acaso; FFT deu 51.9)")


main()

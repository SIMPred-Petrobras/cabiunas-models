#!/usr/bin/env python3
"""EXP 42 — OS QUATRO CANAIS DE HOJE E O QUE O VOTO FAZ COM ELES (descritivo; 8 eventos; sem hipótese nova).

REGISTRO (17/10/2026). Descritivo, pedido pelo usuário ('quais canais temos hoje? como usar o voto para reduzir os FP?'): nenhum critério de ganho, nenhum parâmetro escolhido pelo resultado. As contagens por padrão de canais, com 9 FP e 9 episódios de trip (8 eventos), servem para ORIENTAR o desenho do EXP 43, não para concluir.
O QUE SE MEDE (sinais de produção de 8 eventos; `ci.prever` intocado): (1) para cada canal, o tempo vigiado em que ele está aceso no nível A e no nível B; (2) desligando um canal por vez (k enorme), a detecção e os FP do detector inteiro; (3) para os 18 episódios, quais canais estão acesos no NASCIMENTO (níveis A e B) e as contagens por padrão para FP e para trips.

RESULTADO (17/10/2026) -- descritivo.
  OS QUATRO CANAIS (parâmetros de produção): t = resíduo do PCA (máximo por sensor, normalizado pelo p99 do bundle) das 14 temperaturas (TI_0325, 0315, 0317, 0305, 0307, 0303, 0301; TC382_01 a 06; T5_AVG_A), meia-vida 1 h; p = resíduo do PCA das 12 pressões (PI_0315, 0319, 0340, 0339, 5134001, 0307, 0308; PDI_0317, 0302, 0338, 0301; PDIT_0305), meia-vida 1 h; sp = |z| do SPREAD de mancal (TI_0305 menos a mediana de TI_0301, 0303 e 0307, contra a mediana 10,63 e o MAD 0,515 do bundle), meia-vida 30 min; vb = z-score da vibração das 10 sondas TV_351 a 355 X/Y contra uma referência rolante de 400 h que exclui os trips conhecidos (7 dias), meia-vida 30 min. Limiares base x k (nível A / B): t 2,20 / 3,40; p 2,40 / 3,40; sp 2,70 / 5,10; vb 6,00 / 6,60. VOTO: A = 3 de 4; B = 2 de 4 com sp ou vb; CUSUM por canal (kappa 0,75, h 80, carga_reset 0,25), refratário 72 h, duração mínima 120 min.
  (1) TEMPO VIGIADO ACESO (nível A / B): t 34,9% / 25,0%; p 35,2% / 31,6%; sp 59,6% / 43,5%; vb 59,5% / 51,2%: o sp e o vb ficam acesos 43 a 60% do tempo (são ESTADOS); t e p, 25 a 35%.
  (2) DESLIGANDO UM CANAL (detecção, FP): sem t 3/8, FP 15 (perde 4 eventos e SOBEM os FP); sem p 5/8, FP 9 (perde 27/02/2025 e 26/02/2026 e NÃO reduz FP); sem sp 5/8, FP 12 (perde 17/03 e 11/04 e SOBEM os FP: o sp filtra FP); sem vb 2/8, FP 8 (perde seis eventos). Nenhum canal é 'só FP': cada um participa das detecções.
  (3) CANAIS NO NASCIMENTO (nível B): FP: p+vb x3, t+p+vb x2, t+sp+vb x2 (sem p), p+sp x1, p+sp+vb x1: p está em 7 dos 9 FP; trips: p+vb x3 (27/02/2025 duas vezes e a corrida de 647 h), só vb x3 (07/04, 11/04, 09/12, nascidos pelo A), t+vb x1, p+sp x1, só p x1. O par p+vb nasce em 3 FP (16/06/2025, 16/01/2026, 08/04/2026) e em UM evento de trip (27/02/2025: 140 h e 1,6 h).
LEITURA. Os quatro canais participam das detecções; sp e vb são estados quase sempre acesos e quem decide o voto são t e p; o p aparece em 7 de 9 FP, mas tirá-lo NÃO reduz o FP (9 para 9) e custa 2 detecções; o par p+vb é o padrão mais comum dos FP só-B e também o do trip de 27/02/2025.

Uso:  PYTHONPATH=src .venv/bin/python experiments/42_canais_e_votacao_hoje.py
"""
from __future__ import annotations
import contextlib, copy, io, warnings
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48); CAN = ("t", "p", "sp", "vb")


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)

    def roda(d):
        with contextlib.redirect_stdout(io.StringIO()):
            return ci.prever({"detector": d}, proc)
    R = roda(det); mask = R["vigiado"].astype(bool); en = (R.index >= INICIO8) & mask.to_numpy(); n = int(en.sum())
    ev = lambda al: avalia(al.astype(bool).reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8); r0 = ev(R["is_anomaly"])
    print(f"PUBLICADO: {r0.det}/8, FP {r0.fp}, {r0.horas_alarme:.0f} h\n\n(1) TEMPO VIGIADO COM CADA CANAL ACESO (a partir de 01/02/2025)", flush=True)
    for c in CAN:
        print(f"   {c:2s}: nível A {100 * R[f'a_{c}'].to_numpy()[en].mean():5.1f}% (limiar base x k = {det['base'][c] * det['k_nivel_a'][c]:.2f}) | nível B {100 * R[f'b_{c}'].to_numpy()[en].mean():5.1f}% (limiar {det['base'][c] * det['k_nivel_b'][c]:.2f}) | meia-vida {det['halflife'][c]}", flush=True)
    print("\n(2) DESLIGANDO UM CANAL POR VEZ (k enorme)", flush=True)
    for c in CAN:
        d = copy.deepcopy(det); d["k_nivel_a"][c] = 1e6; d["k_nivel_b"][c] = 1e6; r = ev(roda(d)["is_anomaly"])
        print(f"   sem {c:2s}: {r.det}/8, FP {r.fp}, {r.horas_alarme:.0f} h | eventos detectados: { {k[:10]: x['det'] for k, x in r.por_evento.items()} }", flush=True)
    print("\n(3) CANAIS ACESOS NO NASCIMENTO (A/B), por padrão; episódios do publicado", flush=True); eps = [(a, b) for a, b in episodios_de_serie(R["is_anomaly"].astype(bool).loc[INICIO8:])]; linhas = []
    for a, b in eps:
        tr = [t for t in reg if a <= t and b >= t - JAN]; A = "".join(c for c in CAN if R.loc[a, f"a_{c}"]) or "-"; B = "".join(c for c in CAN if R.loc[a, f"b_{c}"]) or "-"
        linhas.append(dict(inicio=f"{a:%Y-%m-%d %H:%M}", dur_h=round((b - a).total_seconds() / 3600 + 2 / 60, 1), tipo=("trip " + str(tr[0])[:10]) if tr else "FP", A=A, B=B))
    T = pd.DataFrame(linhas); print(T.to_string(index=False), flush=True)
    for tipo in ("FP", "trip"):
        sub = T[T.tipo.str.startswith(tipo)]; print(f"   {tipo}: padrões B no nascimento: {dict(Counter(sub.B))} | padrões A: {dict(Counter(sub.A))}", flush=True)


if __name__ == "__main__":
    main()

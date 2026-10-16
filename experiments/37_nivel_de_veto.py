#!/usr/bin/env python3
"""EXP 37 — UM TERCEIRO ELEMENTO DE 'VETO' (supressão baseada em estado): se a condição estiver ligada, o alarme é desconsiderado.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. Ideia do usuário: uma terceira vertente que, quando ligada, desconsidera o alarme. Em gestão de alarmes é a SUPRESSÃO BASEADA EM ESTADO (ISA-18.2): condições em que o alarme não tem significado. O EXP 36 mostrou que os FP passam pelo nível B e que os FP só-B têm o mesmo padrão de canais de um trip (p + vb): um veto desenhado olhando os 9 FP seria ajustado aos mesmos eventos. Por isso as condições abaixo são fixadas AGORA, por razão física ou de qualidade de dado, e os limiares vêm da distribuição de um período ANTERIOR (nov/2024 a jan/2025), sem olhar nenhum FP ou trip.
IMPLEMENTAÇÃO. O veto suspende a vigilância (a máscara de produção é fechada e o CUSUM zerado nos instantes de veto), que é o mecanismo de blackout do v2; `ci.prever` é a caixa-preta; sinais de 8 eventos; INICIO = 01/02/2025.
CONDIÇÕES (fixadas): V_T TRANSIENTE DE CARGA: |T5(t) - T5(t - 60 min)| >= o percentil 95 dessa variação nos instantes vigiados de 01/11/2024 a 31/01/2025, com histerese de 30 min (o veto continua 30 min depois). V_F SENSORES TRAVADOS: a fração das 26 tags de t e p cujo desvio-padrão nos últimos 30 min é zero >= o percentil 95 dessa fração no mesmo período anterior, com histerese de 30 min.
MEDIDAS: duty do veto (fração do tempo vigiado suspenso), detecção de 8, FP, horas totais; quais episódios do publicado somem (FP e trip). PERMUTAÇÃO: 100 vetos ALEATÓRIOS com o mesmo duty (blocos de comprimento igual à mediana dos blocos do veto real, posições uniformes no período vigiado): a distribuição de (detecção, FP) ao acaso; o veto real 'remove FP além do acaso' se o seu FP é <= o percentil 10 dos aleatórios E a detecção >= 7/8.
CRITÉRIOS (fixados antes). Um veto é ÚTIL se: detecção >= 7/8 (a do publicado), FP <= 7 (-2) e remove FP além do acaso (FP no percentil <= 10 e detecção >= o percentil 50 dos aleatórios). É NOCIVO se a detecção cai >= 1 evento sem cortar FP. Nenhum limiar é ajustado depois de ver o resultado; o ruído com 8 eventos é de +-2 (só vale uma diferença visível no placebo).
EXPECTATIVAS: V_T custa >= 1 detecção (as partidas e os transientes são onde caem os trips: EXP 02/03) ~65% e corta <= 1 FP ~70%; V_F tem duty <= 2% e não muda o resultado ~65%; algum veto ÚTIL ~8%.
INTERPRETAÇÃO PREVISTA. Um veto só se justifica com uma condição física que o operador reconheça (gás combustível instável, lavagem, calibração de instrumento, rampa de carga): a conversa com a operação sobre quais estados geram alarme sem significado é a fonte de condições a priori melhores do que as destas duas. Este experimento só mede se as duas candidatas, derivadas dos dados disponíveis, têm utilidade.
LIMITES. 8 eventos e 9 FP; duas condições; o veto reaproveita a suspensão de vigilância (que zera o CUSUM) e não suprime só o voto; o período de referência dos limiares é curto (3 meses); nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/37_nivel_de_veto.py
"""
from __future__ import annotations
import contextlib, io, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48); GL = {}


def prever_com_veto(veto):
    proc = GL["proc"].copy(); v = np.asarray(veto, bool); proc["mask"] = proc["mask"].to_numpy(bool) & ~v; proc["reset"] = proc["reset"].to_numpy(bool) | v
    with contextlib.redirect_stdout(io.StringIO()):
        R = GL["ci"].prever({"detector": GL["det"]}, proc)
    return R["is_anomaly"].astype(bool)


def mede(al):
    r = avalia(al.reindex(GL["al_c"].index).fillna(False), GL["vig"], GL["reg"], inicio_periodo=INICIO8); return r


def tarefa(a):
    i, veto = a; r = mede(prever_com_veto(veto)); return i, (r.det, r.fp, r.horas_alarme)


def dilata(x, n):
    s = pd.Series(x.astype(float)); return (s.rolling(n + 1, min_periods=1).max().to_numpy() > 0)


def blocos_aleatorios(rng, n, duty, comp, mask):
    alvo = int(duty * mask.sum()); v = np.zeros(n, bool); usado = 0; pos = np.flatnonzero(mask)
    while usado < alvo:
        i = int(rng.choice(pos)); j = min(i + comp, n); v[i:j] |= mask[i:j]; usado = int((v & mask).sum())
    return v


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = list(modelos[-1]["normalizacao"]["temperatura"]["cols"]) + list(modelos[-1]["normalizacao"]["pressao"]["cols"])
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index
    g = nvio.carrega_grade().reindex(idx); GL.update(ci=ci, det=det, proc=proc, reg=reg, al_c=al_c, vig=vig.reindex(idx).fillna(False))
    mask = proc["mask"].to_numpy(bool); ref = (idx >= pd.Timestamp("2024-11-01", tz="UTC")) & (idx < pd.Timestamp("2025-02-01", tz="UTC")) & mask
    dT5 = (g["T5_AVG_A"] - g["T5_AVG_A"].shift(30)).abs().to_numpy(); thr_T = float(np.nanquantile(dT5[ref], 0.95))
    plano = (g[tags].rolling(15, min_periods=15).std() == 0).mean(axis=1).to_numpy(); thr_F = float(np.nanquantile(plano[ref], 0.95))
    vetos = {"V_T transiente de carga": dilata(np.nan_to_num(dT5, nan=0) >= thr_T, 15), "V_F sensores travados": dilata(np.nan_to_num(plano, nan=0) >= max(thr_F, 1e-9) if thr_F > 0 else np.zeros(len(idx), bool), 15)}
    print(f"limiares (período 01/11/2024-31/01/2025): V_T |dT5 60 min| >= {thr_T:.1f} °C; V_F fração de tags travadas >= {thr_F:.3f}", flush=True)
    pub = mede(prever_com_veto(np.zeros(len(idx), bool)))
    lin = lambda r: f"{r.det}/8, início {r.inicio}, FP {r.fp} ({r.fp_mes:.2f}/mês), totais {r.horas_alarme:.0f} h"
    print(f"PUBLICADO: {lin(pub)}", flush=True)
    ep_pub = episodios_de_serie(prever_com_veto(np.zeros(len(idx), bool)).loc[INICIO8:]); rng = np.random.default_rng(20261016)
    for nome, v in vetos.items():
        duty = float((v & mask & (idx >= INICIO8)).sum() / max((mask & (idx >= INICIO8)).sum(), 1)); al = prever_com_veto(v); r = mede(al); ep = episodios_de_serie(al.loc[INICIO8:])
        somem = [(a, b) for a, b in ep_pub if not any(a <= d and b >= c for c, d in ep)]; tipo = lambda a, b: ("trip" if any(a <= t and b >= t - JAN for t in reg) else "FP")
        print(f"\n{nome}: duty {100 * duty:.2f}% do tempo vigiado | {lin(r)} | somem do publicado: {[f'{a:%Y-%m-%d} {tipo(a, b)}' for a, b in somem]}", flush=True)
        if duty <= 0:
            continue
        runs = []; s = (v & mask).astype(int); d = np.diff(np.concatenate(([0], s, [0]))); comp = int(np.median(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))) if (d == 1).any() else 15; comp = max(comp, 1)
        vs = [blocos_aleatorios(rng, len(idx), duty, comp, mask & (idx >= INICIO8)) for _ in range(100)]
        with Pool(4) as p:
            R = [x[1] for x in p.map(tarefa, list(enumerate(vs)), chunksize=4)]
        dets = np.array([x[0] for x in R]); fps = np.array([x[1] for x in R])
        pf = 100 * float(np.mean(fps < r.fp)); pd_ = 100 * float(np.mean(dets < r.det))
        print(f"   permutação (100 vetos aleatórios, mesmo duty, blocos de {comp} passos): detecção mediana {np.median(dets):.0f}/8 (mín {dets.min()}, máx {dets.max()}), FP mediano {np.median(fps):.0f} (mín {fps.min()}, máx {fps.max()}) | o veto real tem FP menor que {pf:.0f}% dos aleatórios e detecção menor que {pd_:.0f}% deles", flush=True)
        util = r.det >= 7 and r.fp <= 7 and pf >= 90 and pd_ <= 50; nocivo = r.det <= pub.det - 1 and r.fp >= pub.fp
        print(f"   -> {'ÚTIL' if util else 'NOCIVO' if nocivo else 'NEUTRO/INCONCLUSIVO'}", flush=True)


if __name__ == "__main__":
    main()

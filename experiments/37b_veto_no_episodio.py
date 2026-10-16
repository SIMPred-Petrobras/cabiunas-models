#!/usr/bin/env python3
"""EXP 37b — O VETO COMO SUPRESSÃO DO EPISÓDIO (sem fechar a vigilância): o episódio do publicado é desconsiderado se a condição de veto está ligada no seu nascimento.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR. (Correção de desenho do EXP 37: lá o veto fechava a máscara e zerava o CUSUM, e vetos ALEATÓRIOS do mesmo duty davam mediana de 18 FP contra 9 do publicado, porque cada reabertura da máscara cria alarmes de borda: o teste media o artefato do blackout, não a ideia.)

O QUE SE FAZ. Os mesmos 18 episódios do detector publicado (sinais de 8 eventos, `ci.prever` sem alteração) e as MESMAS duas condições de veto, com os mesmos limiares do período anterior (nov/2024 a jan/2025; EXP 37): V_T transiente de carga (|T5(t) - T5(t - 60 min)| >= 17,6 °C, com histerese de 30 min) e V_F sensores travados (fração de tags de t e p com desvio-padrão zero em 30 min >= 0,038, com histerese de 30 min). Um episódio é VETADO se a condição está ligada em algum instante da janela [nascimento - 30 min, nascimento + 30 min]. Medem-se detecção de 8, FP e horas fora; e qual episódio some (FP ou trip).
PERMUTAÇÃO: 2.000 vetos aleatórios com o mesmo duty e o mesmo comprimento mediano de bloco, e o mesmo critério de veto no nascimento: a distribuição, ao acaso, do número de FP e de trips vetados. O veto 'remove FP além do acaso' se o número de FP vetados é >= o percentil 90 dos aleatórios e o de trips vetados <= a mediana dos aleatórios.
CRITÉRIOS (fixados antes). ÚTIL = detecção >= 7/8, FP <= 7 e remove FP além do acaso; NOCIVO = detecção <= 6/8 sem FP <= 7.
EXPECTATIVAS: V_T veta >= 1 trip ~55% (as partidas e os transientes são onde caem os trips) e <= 1 FP ~65%; V_F, com duty de ~30% no período avaliado, veta >= 3 episódios ao acaso e o seu efeito não se distingue de aleatório ~70%; algum ÚTIL ~6%.
LIMITES. Os 9 FP e 9 episódios de trip; as duas condições e os limiares fixados no EXP 37 (que não se ajustam aqui); o veto no nascimento ignora que um episódio pode ter o transiente só depois; nada se adota sem sombra.

RESULTADO (16/10/2026) -- NENHUMA DAS DUAS CONDIÇÕES É UM BOM VETO: o V_T veta trips, o V_F é muito frequente e não é seletivo.
  (Desvio do pré-registro: com duty de 30% a permutação de 2.000 sorteios estourou o tempo; para o V_F usaram-se 300 sorteios. O V_T usou os 2.000 previstos.)
  PUBLICADO: 7/8, 9 FP, 18 episódios (9 FP, 9 de trip). V_T (duty 3,86%): veta 2 de 18 episódios, AMBOS trips (11/04/2025 e o de 27/04/2025, onset do trip de 29/04) e nenhum FP: 5/8, 9 FP, 407 h; os aleatórios vetam, mediana, 0 FP e 0 trips (p90 1 e 1), e P(aleatório veta >= 2 trips) = 0,086: o V_T veta trips MAIS que o acaso e nenhum FP: NOCIVO. V_F (duty 30,65%): veta 4 episódios (3 FP: 31/03/2025, 16/06/2025, 02/04/2026; e 1 trip: o de 09/03, onset do trip de 17/03): 6/8, 6 FP, 324 h; os aleatórios do mesmo duty vetam, mediana, 3 FP (p90 5) e 4 trips (p90 6): P(aleatório veta >= 3 FP) = 0,70, logo não remove FP além do acaso; veta menos trips que o acaso (1 contra mediana 4), mas com 30,7% do tempo vigiado em veto (cegueira de quase um terço do tempo): NEUTRO/INCONCLUSIVO e operacionalmente inaceitável.
LEITURA. (1) A ideia de um 'terceiro elemento de veto' é COERENTE (supressão baseada em estado, ISA-18.2), mas a condição não pode sair dos FP: os FP só-B têm o padrão de canais de trips (EXP 36), e as duas condições a priori derivadas dos dados disponíveis falham: o transiente de carga é justamente onde caem os trips (partidas, EXP 02/03) e os sensores travados são frequentes demais (o limiar do período anterior não se transfere: 5% viram 30,7% do tempo). (2) Um veto implementado como suspensão da vigilância cria alarmes de borda (os vetos aleatórios dobram os FP): se existir uma condição útil, deve agir sobre o alarme/episódio. (3) A fonte certa de condições é a OPERAÇÃO: estados conhecidos em que o alarme não tem significado (gás combustível instável, lavagem, calibração de instrumento, mudança de combustível); sem essa informação, este tipo de veto fica sem condição de teste coerente.
PLACAR: ACERTEI: V_T veta >= 1 trip (~55%: vetou 2) e corta <= 1 FP (~65%: 0); V_F não se distingue do acaso (~70%); nenhum ÚTIL (~94%). PARCIAL: V_F veta >= 3 episódios ao acaso (~70%: 4 vetados, mediana aleatória 3 FP e 4 trips).

Uso:  PYTHONPATH=src .venv/bin/python experiments/37b_veto_no_episodio.py
"""
from __future__ import annotations
import contextlib, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E37 = _carrega("exp37", "37_nivel_de_veto.py"); GL = E37.GL
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

INICIO8 = E37.INICIO8; JAN = E37.JAN; ROOT = E37.ROOT


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = list(modelos[-1]["normalizacao"]["temperatura"]["cols"]) + list(modelos[-1]["normalizacao"]["pressao"]["cols"])
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index
    g = nvio.carrega_grade().reindex(idx); vig = vig.reindex(idx).fillna(False); GL.update(ci=ci, det=det, proc=proc, reg=reg, al_c=al_c, vig=vig)
    mask = proc["mask"].to_numpy(bool); ref = (idx >= pd.Timestamp("2024-11-01", tz="UTC")) & (idx < pd.Timestamp("2025-02-01", tz="UTC")) & mask
    dT5 = (g["T5_AVG_A"] - g["T5_AVG_A"].shift(30)).abs().to_numpy(); thr_T = float(np.nanquantile(dT5[ref], 0.95)); plano = (g[tags].rolling(15, min_periods=15).std() == 0).mean(axis=1).to_numpy(); thr_F = float(np.nanquantile(plano[ref], 0.95))
    vetos = {"V_T transiente de carga": E37.dilata(np.nan_to_num(dT5, nan=0) >= thr_T, 15), "V_F sensores travados": E37.dilata(np.nan_to_num(plano, nan=0) >= thr_F, 15)}
    with contextlib.redirect_stdout(io.StringIO()):
        R = ci.prever({"detector": det}, proc)
    al = R["is_anomaly"].astype(bool); eps = episodios_de_serie(al.loc[INICIO8:]); tipo = [("trip" if any(a <= t and b >= t - JAN for t in reg) else "FP") for a, b in eps]
    pub = avalia(al.reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8); print(f"PUBLICADO: {pub.det}/8, FP {pub.fp}, {len(eps)} episódios ({tipo.count('FP')} FP, {tipo.count('trip')} trip)", flush=True)
    nasc = np.array([idx.searchsorted(a) for a, b in eps]); rng = np.random.default_rng(20261016); en = (idx >= INICIO8)
    for nome, v in vetos.items():
        duty = float((v & mask & en).sum() / (mask & en).sum()); vet = np.array([bool(v[max(i - 15, 0):i + 16].any()) for i in nasc]); fpv = int(sum(1 for x, t in zip(vet, tipo) if x and t == "FP")); trv = int(sum(1 for x, t in zip(vet, tipo) if x and t == "trip"))
        al2 = al.copy()
        for (a, b), x in zip(eps, vet):
            if x:
                al2.loc[a:b] = False
        r = avalia(al2.reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8)
        print(f"\n{nome}: duty {100 * duty:.2f}% | vetados {int(vet.sum())} de {len(eps)} episódios: FP {fpv}, trip {trv} -> {r.det}/8, FP {r.fp}, fora {r.horas_alarme_fora:.0f} h | episódios vetados: {[f'{a:%Y-%m-%d} {t}' for (a, b), t, x in zip(eps, tipo, vet) if x]}", flush=True)
        s = (v & mask).astype(int); d = np.diff(np.concatenate(([0], s, [0]))); comp = max(int(np.median(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))), 1) if (d == 1).any() else 15; pos = np.flatnonzero(mask & en); fps = []; trs = []
        for _ in range(2000 if duty < 0.10 else 300):
            vv = np.zeros(len(idx), bool); usado = 0; alvo = duty * len(pos)
            while usado < alvo:
                i = int(rng.choice(pos)); vv[i:i + comp] = True; usado = int((vv & mask & en).sum())
            vt = np.array([bool(vv[max(i - 15, 0):i + 16].any()) for i in nasc]); fps.append(sum(1 for x, t in zip(vt, tipo) if x and t == "FP")); trs.append(sum(1 for x, t in zip(vt, tipo) if x and t == "trip"))
        fps = np.array(fps); trs = np.array(trs)
        print(f"   permutação (2.000 vetos aleatórios, mesmo duty, blocos de {comp} passos): FP vetados mediana {np.median(fps):.0f} (p90 {np.quantile(fps, .9):.0f}); trips vetados mediana {np.median(trs):.0f} (p90 {np.quantile(trs, .9):.0f}) | P(aleatório veta >= {fpv} FP) = {np.mean(fps >= fpv):.3f}; P(aleatório veta >= {trv} trips) = {np.mean(trs >= trv):.3f}", flush=True)
        alem = fpv >= np.quantile(fps, 0.9) and trv <= np.median(trs); util = r.det >= 7 and r.fp <= 7 and alem; nocivo = r.det <= 6 and not (r.fp <= 7)
        print(f"   -> {'ÚTIL' if util else 'NOCIVO' if nocivo else 'NEUTRO/INCONCLUSIVO'} (remove FP além do acaso: {alem})", flush=True)


if __name__ == "__main__":
    main()

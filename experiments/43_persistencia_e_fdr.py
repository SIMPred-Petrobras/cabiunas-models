#!/usr/bin/env python3
"""EXP 43 — (A) A PERSISTÊNCIA COMO CURVA DE CUSTO E (B) O FDR ENTRE CANAIS NO LUGAR DA CONTAGEM DE LIMIARES.

PRÉ-REGISTRO (17/10/2026), COMMITADO ANTES DE RODAR.

POR QUE (a leitura da literatura, 17/10; só resumos, quase tudo eólica): (1) a persistência deve ser tratada como variável de decisão EXPLÍCITA, e a relação com o FP e a latência é uma curva (persistência moderada corta FP; excessiva atrasa e perde falhas lentas); (2) o procedimento de Benjamini-Hochberg (FDR) entre sensores/canais reduz alarmes multissensor, e um voto 'k de n' supõe canais independentes e de igual taxa, o que o v2 viola (sp e vb acesos 43 a 60% do tempo).
PARTE A. O detector de produção (8 eventos; `ci.prever` intocado) com a duração mínima do episódio (`duracao_min`; a `duracao_min_forte` = min(60, duracao_min)) em {0, 30, 60, 90, 120*, 180, 240, 360} min. Reporta-se, por valor: detecção de 8, início, FP, FP/mês, horas totais e a ANTECEDÊNCIA média (lead) dos eventos detectados. Critério (fixado antes): a persistência é uma ALAVANCA se algum valor diferente de 120 min dá detecção >= 7/8 com FP <= 7 e os dois vizinhos imediatos da grade não perdem >= 2 detecções; do contrário a curva só mostra a troca.
PARTE B. Probabilidade de cauda causal por canal (a de `nv`: referência dos W = 60 dias vigiados anteriores ao mês), p_c(t); a cada instante o procedimento BH sobre os 4 p-valores no nível q: R_t = o maior i com p_(i) <= i q / 4; decisão d_t = (R_t >= r_min), r_min em {2, 3} ('pelo menos r_min canais rejeitados sob FDR q'). Duas formas de persistência sobre d_t: (i) episódio = corrida contínua de d_t por >= 2 h, refratário de 72 h (a política fixa de `nv.politica`); (ii) janela: >= 80% de d_t = 1 nas últimas 2 h, depois a mesma duração e refratário. O nível q é escolhido, para cada (r_min, forma), pela grade logarítmica de 25 valores entre 1e-4 e 0,5 que dá o TOTAL de horas de alarme mais perto do do publicado (1.457 h), SEM rótulos. Medem-se detecção de 8, FP e horas; placebo (eventos deslocados); auditoria do futuro (sinais embaralhados depois de 01/09/2025).
COMPARAÇÃO: o voto de produção ao mesmo custo total (7/8, 9 FP). CRITÉRIOS (fixados antes): o FDR EQUIVALE se detecção >= 6/8 e FP <= 11 ao custo total igual; GANHA se detecção >= 7/8 e FP <= 7; PIOR o resto.
EXPECTATIVAS: A: a curva é plana entre 60 e 180 min em detecção (7/8 em >= 3 valores) ~50%; alguma ALAVANCA ~10%; B: o FDR EQUIVALE ~20%, GANHA ~5%; o r_min = 2 melhor que 3 em detecção ~60%; o FDR com persistência em janela melhor que o episódio contínuo ~45%; a auditoria passa ~95%.
INTERPRETAÇÃO PREVISTA. A: se não há alavanca, a persistência atual (120 min) está num platô e o FP não se reduz por ela sem custo. B: se o FDR chegar perto do voto com dois parâmetros (q, W), é uma alternativa mais honesta (menos ajuste à mão) e o primeiro desenho 'de princípio' que sobrevive; se ficar muito abaixo (como a fusão de Fisher: 6/8 com 42 FP sem o CUSUM), o voto de produção carrega a informação do CUSUM por canal e dos dois níveis.
LIMITES. 8 eventos; a fusão do FDR aqui é instantânea (sem o CUSUM por canal, para isolar a regra de decisão); a referência dos 60 dias anteriores contém alarmes; os p-valores de canais dependentes violam a premissa do BH (a garantia de FDR não vale, o BH é usado como regra de decisão calibrada); nada se adota sem sombra.

RESULTADO (17/10/2026) -- A: NENHUMA ALAVANCA NA PERSISTÊNCIA (o 120 min está no joelho da curva); B: O FDR INSTANTÂNEO É MUITO PIOR (36 a 41 FP) E NÃO ALCANÇA O CUSTO.
  PARTE A (duracao_min -> detecção de 8, início, FP, horas totais, lead): 0 min: 8/8, 7, 18 FP, 1.464 h, 31,6 h | 30 min: 8/8, 7, 16 FP, 1.464 h | 60 min: 7/8, 6, 11 FP, 1.459 h, 34,9 h | 90 min: 7/8, 6, 9 FP, 1.457 h | 120 min* (produção): 7/8, 6, 9 FP, 1.457 h, 34,9 h | 180 min: 6/8, 5, 9 FP, 1.454 h | 240 min: 6/8, 5, 9 FP | 360 min: 5/8, 4, 9 FP, 1.450 h, 40,6 h. O 8.º evento (04/11/2025, cujo voto dura ~46 min) só é detectado com duracao_min <= 30 min, a +7 FP (16 a 18 contra 9). A partir de 90 min o FP não cai mais (9) e a detecção cai a 6/8 em 180 min e a 5/8 em 360: o 120 min é o joelho. ALAVANCA (>= 7/8 com FP <= 7 fora de 120 min): nenhuma.
  PARTE B (BH entre os 4 canais, probabilidades de cauda causais de 60 dias; custo total-alvo 1.457 h sem rótulos; AUDITORIA DO FUTURO: diferença 0, PASSA): em todas as quatro configurações o q escolhido ficou no TETO da grade (0,5) e as horas totais não chegaram a 1.457 h (979, 1.019, 709 e 635 h): r_min = 2, episódio contínuo: 7/8, início 5, 41 FP (3,68/mês), 979 h, placebo ~3,2/8; r_min = 2, janela 80% em 2 h: 7/8, início 6, 36 FP (3,23/mês), 1.019 h; r_min = 3, episódio contínuo: 5/8, início 3, 38 FP, 709 h; r_min = 3, janela: 4/8, início 3, 32 FP, 635 h, placebo ~2,0/8. Todas PIOR (o voto de produção ao mesmo custo: 7/8, 9 FP).
LEITURA. (1) A persistência é, como a literatura recomenda, uma variável de decisão explícita, e a curva é clara: 120 min é o joelho (abaixo dele os FP sobem; acima, a detecção cai sem reduzir FP); o 8.º evento custa 7 FP (16 contra 9) com persistência <= 30 min. Isso é uma escolha de MODO, não uma melhoria: sensível (30 min: 8/8, 16 FP, 1,43/mês) contra conservador (120 min: 7/8, 9 FP, 0,81/mês); a leitura que a literatura sugere para operar os dois é em DOIS NÍVEIS de notificação (pré-alerta mais sensível e alerta confirmado), não um único limiar. (2) O FDR (BH) aplicado instante a instante, sem acúmulo por canal, produz 32 a 41 episódios de FP: como a fusão de Fisher sem CUSUM (EXP 35, 42 FP) e como o 'quinto canal' vigiado: regras de decisão instantâneas geram dezenas de episódios curtos; o CUSUM por canal do v2 é o que mantém os FP em 9. A grade de q parou no teto (0,5), o que significa que o custo-alvo não foi alcançado pelo BH, e a premissa de FDR não vale com canais dependentes. (3) Não há aqui um desenho 'de princípio' que supere o voto de produção.
PLACAR: ACERTEI: a curva é plana entre 60 e 180 min em detecção (7/8 em 60, 90 e 120; ~50%); nenhuma alavanca (~90%); o FDR não EQUIVALE nem GANHA (~75% e ~95%); r_min = 2 melhor que 3 em detecção (~60%); a auditoria passa (~95%). ERREI: o FDR com janela de 80% melhor que o episódio contínuo (~45%: iguais em detecção; FP 36 contra 41).

Uso:  PYTHONPATH=src .venv/bin/python experiments/43_persistencia_e_fdr.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E35 = _carrega("exp35", "35_fusao_probabilistica.py"); E34 = E35.E34
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG
from nv.politica import episodios, alarme_de

ROOT = E35.ROOT; INICIO8 = E35.INICIO8; CANAIS = E35.CANAIS; TOTAL = 1457.0


def bh_R(U, q):
    P = np.column_stack([U[c] for c in CANAIS]); ok = np.isfinite(P).all(axis=1); S = np.sort(np.where(np.isfinite(P), P, 1.0), axis=1); lim = (np.arange(1, 5) * q / 4.0)[None, :]; cond = S <= lim
    R = np.where(cond.any(axis=1), 4 - np.argmax(cond[:, ::-1], axis=1), 0); return np.where(ok, R, 0)


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index; vig = vig_c.reindex(idx).fillna(False)
    lin = lambda r: f"{r.det}/8, início {r.inicio}, FP {r.fp} ({r.fp_mes:.2f}/mês), totais {r.horas_alarme:.0f} h, lead {r.lead_medio_h:.1f} h"
    print("PARTE A — PERSISTÊNCIA (duração mínima do episódio) COMO CURVA DE CUSTO", flush=True); res = {}
    for dm in (0, 30, 60, 90, 120, 180, 240, 360):
        d = copy.deepcopy(det); d["duracao_min"] = dm; d["duracao_min_forte"] = min(60, dm)
        with contextlib.redirect_stdout(io.StringIO()):
            R = ci.prever({"detector": d}, proc)
        r = avalia(R["is_anomaly"].astype(bool).reindex(al_c.index).fillna(False), vig_c.reindex(al_c.index).fillna(False), reg, inicio_periodo=INICIO8); res[dm] = r
        print(f"   duracao_min = {dm:3d} min{'*' if dm == 120 else ' '}: {lin(r)}", flush=True)
    alav = [dm for dm in res if dm != 120 and res[dm].det >= 7 and res[dm].fp <= 7]
    print(f"   ALAVANCA (detecção >= 7/8 e FP <= 7 em um valor diferente de 120 min): {alav if alav else 'nenhuma'}", flush=True)
    print("\nPARTE B — FDR (BH) ENTRE OS 4 CANAIS, custo TOTAL de 1.457 h sem rótulos", flush=True)
    corte = pd.Timestamp("2025-09-01", tz="UTC"); p2 = proc.copy(); pos = idx >= corte; p2.loc[pos, list(CANAIS)] = p2.loc[pos, list(CANAIS)].sample(frac=1.0, random_state=1).to_numpy(); sl = (idx >= INICIO8) & (idx < corte)
    Ua = E35.caudas(proc, 60); Ub = E35.caudas(p2, 60); d1 = max(float(np.nanmax(np.abs(Ua[c][sl] - Ub[c][sl]))) for c in CANAIS); print(f"   AUDITORIA DO FUTURO: diferença máxima das probabilidades de cauda antes de 01/09/2025 = {d1:.2e} -> {'PASSA' if d1 < 1e-9 else 'FALHA'}", flush=True)
    U = Ua; qs = np.logspace(-4, np.log10(0.5), 25)
    for rmin in (2, 3):
        for forma in ("episódio contínuo", "janela 80%/2 h"):
            melhor = None
            for q in qs:
                dt = (bh_R(U, q) >= rmin).astype(float); s = pd.Series(dt, index=idx)
                if forma.startswith("janela"):
                    s = (s.rolling(60, min_periods=60).mean() >= 0.8).astype(float)
                al = alarme_de(episodios(s, 0.5), idx); r = avalia(al, vig, reg, inicio_periodo=INICIO8); gap = abs(r.horas_alarme - TOTAL)
                if melhor is None or gap < melhor[0]:
                    melhor = (gap, q, r, al)
            _, q, r, al = melhor; cl = "GANHO" if r.det >= 7 and r.fp <= 7 else "EQUIVALENTE" if r.det >= 6 and r.fp <= 11 else "PIOR"
            print(f"   r_min = {rmin}, {forma:18s}: q = {q:.4f} -> {lin(r)} | placebo ~{E35.placebo(al, vig, reg):.1f}/8 -> {cl}", flush=True)


if __name__ == "__main__":
    main()

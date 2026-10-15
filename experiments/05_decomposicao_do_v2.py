#!/usr/bin/env python3
"""EXP 05 — DE ONDE VEM A VANTAGEM DO v2? Os quatro sinais do v2 sob a política SIMPLES dos baselines.

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. No EXP 04 todos os baselines (escore único + política simples) ficaram em 1 a 4 de 7 eventos contra 6 de 7 do controle, a custo igual. A vantagem do v2 está nos SINAIS (quatro canais por tipo, o `vb` com referência rolante de 400 h, o `sp` do mancal, o PCA por família) ou na ENGENHARIA DA DECISÃO
(CUSUM com memória, dois níveis A/B, portão sp|vb, escalada, voto)? A resposta decide onde a Fase 3 investe: em modelos de sinal ou em modelos/políticas de decisão.

O QUE SE FAZ.
  · Os sinais t, p, sp e vb do v2 (já suavizados pelo EWMA do próprio detector) e a máscara vêm do caminho de PRODUÇÃO, de forma causal (`nv.controle_sinais`: replay semanal, janela de 75 d, registro crescendo, fatiado nos trips). Cada canal é normalizado pelo limiar do nível A do bundle: e_c = sinal_c / (base_c x k_nivel_a_c), então 1 = o limiar sensível do v2.
  · A POLÍTICA SIMPLES do EXP 04 (limiar único, duração >= 2 h, refratário 72 h), sem segundo EWMA (o sinal já é suavizado), sem CUSUM, sem nível B, sem portão, sem escalada. O limiar é calibrado a custo igual (407 h de alarme fora das janelas do controle; 0,5x e 2x como curva) e TAMBÉM no limiar natural (= 1, o do v2).
  · ESCORES: canal_t, canal_p, canal_sp, canal_vb (cada um só); max4 (o maior dos quatro); ord2 (o 2º maior: "pelo menos 2 de 4 acima do limiar"); ord3 (o 3º maior: "pelo menos 3 de 4", o voto do nível A); ord2_spvb = min(max(e_sp, e_vb), ord2) ("pelo menos 2 de 4, um deles sp ou vb", o espírito do nível B).
  · As mesmas métricas e o mesmo harness do EXP 04; 7 eventos.
  · VALIDADE: a máscara extraída tem de reproduzir a vigilância do controle (horas vigiadas dentro de 1%).

EXPECTATIVAS REGISTRADAS (para poder errar de forma visível):
  · Canais sozinhos a 1x: vb 3 a 5 de 7 (o de maior AUC por bloco na pesquisa); sp 1 a 3; t 1 a 3; p 0 a 2.
  · max4 de 3 a 5; ord2 de 4 a 6; ord3 de 3 a 5; ord2_spvb de 4 a 6.
  · Probabilidade de algum destes escores simples chegar a 6/7 a 1x: ~30%; de ord2 ou ord2_spvb chegar a >= 5/7: ~55%.
  · O número de episódios FP dos escores simples a custo igual é maior que o do controle (9): entre 12 e 40.
  · LEITURA PRÉ-REGISTRADA: se o melhor destes escores simples fica em >= 5 de 7, os SINAIS do v2 carregam a maior parte da vantagem; se fica em <= 4 de 7, a CUSTO IGUAL a engenharia da decisão (CUSUM, níveis, portão, escalada) carrega pelo menos 2 eventos.

LIMITES. 7 eventos; a política simples não é a do v2 (que é o controle); o limiar natural (1) e o orçamento são as duas leituras; sem piso de ruído próprio (os sinais não têm aleatoriedade; o piso de ruído do v2 vem do M37 da pesquisa: detecção +-0,5, FP +-0,1); o corte entre "sinal" e "decisão" não é limpo (o EWMA e o `vb` com referência rolante são pré-processamento que também decide).

RESULTADO (15/10/2026). VALIDADE: a máscara extraída pelo caminho de produção tem 7.356 h vigiadas, IGUAL às do controle (+0,00%); 342.029 instantes de 2025-01-01 a 2026-04-21 (39 s). CONTROLE: 6/7, lead 32,7 h, 9 FP, 407 h de orçamento fora das janelas.
  A CUSTO IGUAL (1x; detecção de 7 | início | lead h | FP episódios | horas fora | cobertura ao acaso | p binomial): canal_t 5/7 | 4 | 21,4 | 17 | 446 | 0,132 | 0,0007; canal_p 2/7 | 2 | 14,8 | 24; canal_sp 2/7 | 1 | 36,8 | 12; canal_vb 1/7 | 0 | 48,0 | 11; max4 3/7 | 3 | 22,0 | 20;
        ord2 5/7 | 4 | 23,6 | 23 | 407 | 0,191 | 0,0038; ord3 4/7 | 4 | 10,5 | 30; ord2_spvb 4/7 | 4 | 16,3 | 16 | 323 | 0,139 | 0,0091.
  NO LIMIAR NATURAL DO v2 (nível A, = 1,0; as horas fora das janelas deixam de ser iguais): canal_vb 6/7 | lead 34,0 h | 25 FP | 973 h fora (2,4x o orçamento); max4 6/7 | 28 FP | 2.088 h (5,1x); ord2 5/7 | 31 FP | 612 h; ord3 3/7 | 17 FP | 365 h; canal_t 4/7 | 22 FP | 610 h; canal_sp 3/7 | 1.768 h; canal_p 2/7.
  CURVA: a 0,5x todos os escores dão 0 a 2 de 7; a 2x ord2 chega a 7/7 (43 FP, 689 h fora, lead 26,3 h), ord2_spvb 6/7, ord3 6/7, canal_t 5/7.
  LEITURA PRÉ-REGISTRADA: o melhor escore simples dos sinais do v2 a custo igual fica em 5/7 (canal_t e ord2) contra 1 a 4 dos baselines do EXP 04 e 6 do controle: OS SINAIS DO v2 carregam a maior parte da vantagem (+1 a +4 sobre os baselines), e a engenharia da decisão (CUSUM, níveis, portão, escalada) acrescenta ~1 evento e, principalmente, CONSOLIDA o alarme:
        o controle gasta as 407 h em 9 episódios, a política simples com os mesmos sinais em 16 a 30. O que o v2 tem a mais NÃO é só detecção: é um alarme com MENOS EPISÓDIOS (a métrica que o operador sente) à mesma exposição.
  SURPRESAS: (i) o canal t (PCA das temperaturas) é o melhor canal sozinho a custo igual (5/7, p binomial 0,0007), não o vb: o vb sozinho dá 1/7 a custo igual e só chega a 6/7 com 2,4x o orçamento (alta sensibilidade, baixa especificidade: a mesma leitura do atlas); (ii) o 2º maior canal ("pelo menos 2 de 4") é o melhor combinador a 1x e a 2x (7/7 com 43 FP); o voto de 3 de 4 perde sensibilidade; (iii) o canal p é o pior (2/7).
PLACAR: ACERTEI: o melhor escore simples é >= 5/7 (E de ord2 >= 5: 5/7); o FP dos escores simples fica acima do controle (16 a 30 contra 9); ord2 e ord2_spvb de 4 a 6 (5 e 4). ERREI: o vb como o melhor canal sozinho (é o pior a custo igual: 1/7; previ 3 a 5); o canal t de 1 a 3 (foi 5); max4 de 3 a 5 (3) e ord3 de 3 a 5 (4) ok; nenhum escore simples chega a 6/7 a 1x (previ ~30% de algum: nenhum a 1x; 6/7 no limiar natural do vb e max4, a custos maiores).

Uso:  PYTHONPATH=src .venv/bin/python experiments/05_decomposicao_do_v2.py
"""
from __future__ import annotations
import contextlib, io, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, cobertura_ao_acaso, p_binomial_de_deteccao
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.politica import episodios, alarme_de, calibra_limiar

CACHE = Path(__file__).resolve().parents[1] / "data_local" / "sinais_v2.parquet"
INICIO = pd.Timestamp("2025-03-10", tz="UTC"); CORTE0 = pd.Timestamp("2025-03-05 23:58", tz="UTC")
CANAIS = ("t", "p", "sp", "vb")


def pontua(S, det):
    base = det["base"]; k = det["k_nivel_a"]
    E = pd.DataFrame({c: S[c] / (base[c] * k[c]) for c in CANAIS}).where(S["mask"].astype(bool))
    M = np.sort(np.nan_to_num(E.to_numpy(), nan=-np.inf), axis=1)[:, ::-1]      # maior primeiro; NaN vira -inf (e volta a NaN abaixo)
    M = np.where(np.isfinite(M), M, np.nan)
    out = {f"canal_{c}": E[c] for c in CANAIS}
    out["max4"] = pd.Series(M[:, 0], index=E.index)
    out["ord2"] = pd.Series(M[:, 1], index=E.index)
    out["ord3"] = pd.Series(M[:, 2], index=E.index)
    out["ord2_spvb"] = pd.concat([E[["sp", "vb"]].max(axis=1), out["ord2"]], axis=1).min(axis=1).where(E.notna().all(axis=1))
    return out


def avalia_a_custo(e, vig, reg, alvo_h=None, limiar=None):
    ini = INICIO - pd.Timedelta(days=10); e = e.loc[ini:]; v = vig.loc[ini:]
    thr = limiar if limiar is not None else calibra_limiar(e, v, reg, alvo_h, inicio=INICIO)
    al = alarme_de(episodios(e, thr), e.index)
    return avalia(al, v, reg, inicio_periodo=INICIO), al, thr


def main():
    t0 = pd.Timestamp.now()
    print("extraindo os sinais do v2 de forma causal (replay semanal pelo caminho de produção) ...", flush=True)
    S = sinais_causais(CORTE0, cache=CACHE); print(f"   {len(S)} instantes, {S.index[0]:%Y-%m-%d} a {S.index[-1]:%Y-%m-%d} ({(pd.Timestamp.now() - t0).seconds} s)", flush=True)
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    al_c, vig_c = carrega_controle("atual"); reg = [t for t in carrega_registro() if t >= INICIO]
    ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO); alvo = ctrl.horas_alarme_fora
    vig = S["mask"].astype(bool).reindex(S.index).fillna(False)
    h_extr = float(vig.loc[INICIO:].sum()) * 2 / 60; h_ctrl = ctrl.horas_vigiadas
    print(f"VALIDADE: horas vigiadas na máscara extraída {h_extr:.0f} h contra {h_ctrl:.0f} h do controle ({100 * (h_extr - h_ctrl) / h_ctrl:+.2f}%)")
    print(f"CONTROLE: detecção {ctrl.det}/{ctrl.n_eventos}, lead {ctrl.lead_medio_h:.1f} h, FP {ctrl.fp}, orçamento {alvo:.0f} h fora das janelas")
    P = pontua(S, det); linhas = []
    for nome, e in P.items():
        for rot, kw in (("0,5x", dict(alvo_h=0.5 * alvo)), ("1x", dict(alvo_h=alvo)), ("2x", dict(alvo_h=2 * alvo)), ("natural(1,0)", dict(limiar=1.0))):
            r, al, thr = avalia_a_custo(e, vig, reg, **kw)
            cov = cobertura_ao_acaso(al, vig, reg, inicio_periodo=INICIO)
            linhas.append(dict(escore=nome, ponto=rot, limiar=round(thr, 2), det=f"{r.det}/{r.n_eventos}", inicio=r.inicio, banda=r.banda, lead_h=round(r.lead_medio_h, 1), fp=r.fp, horas=round(r.horas_alarme), fora_h=round(r.horas_alarme_fora), acaso=round(cov, 3), p_binom=round(p_binomial_de_deteccao(r.det, r.n_eventos, cov), 4)))
    T = pd.DataFrame(linhas)
    print("\nA CUSTO IGUAL (1x = 407 h fora das janelas)"); print(T[T.ponto == "1x"].drop(columns="ponto").to_string(index=False))
    print("\nNO LIMIAR NATURAL DO v2 (nível A, = 1,0)"); print(T[T.ponto == "natural(1,0)"].drop(columns="ponto").to_string(index=False))
    print("\nCURVA (0,5x e 2x)"); print(T[T.ponto.isin(["0,5x", "2x"])].to_string(index=False))
    melhor = max(int(x.split("/")[0]) for x in T[T.ponto == "1x"].det)
    print(f"\nMELHOR DETECÇÃO A 1x entre os escores simples dos sinais do v2: {melhor}/{ctrl.n_eventos} (controle {ctrl.det}/{ctrl.n_eventos}) -> " + ("OS SINAIS carregam a maior parte (>= 5)" if melhor >= 5 else "A ENGENHARIA DA DECISÃO carrega pelo menos 2 eventos (<= 4)"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXP 35 — FUSÃO PROBABILÍSTICA DOS QUATRO CANAIS (Fisher + CUSUM) CONTRA O VOTO DE PRODUÇÃO, SEM RÓTULO NA CALIBRAÇÃO E COM AUDITORIA DO FUTURO.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O EXP 33 mostrou que o valor do v2 está na consolidação: o t isolado dá 7/8 com ~18 FP e o voto de produção (3 de 4 no nível A, 2 de 4 com portão mecânico no nível B, CUSUM por canal, refratário, duração) dá 7/8 com 9 FP; e um máximo ingênuo de t e p é muito pior. O voto e os k de nível foram ajustados à mão nos mesmos eventos (EXP 24: o ponto é o topo de um pico). Uma fusão BASEADA EM PRINCÍPIO, com muito menos parâmetros ajustados e nenhum rótulo, é a alternativa coerente.
A FUSÃO (M1). Entrada: os quatro sinais de produção t, p, sp, vb (já suavizados) e a máscara/reset de produção (8 eventos, `sinais_v2_8ev.parquet`). (1) PROBABILIDADE DE CAUDA por canal, CAUSAL: para cada mês, a distribuição de referência do canal é a dos instantes vigiados dos W dias ANTERIORES ao dia 1 (W = 60 de base); u_c = (nº de referências >= s + 1)/(n + 1). (2) FISHER: F = -2 soma_c ln(u_c) (sob normalidade e independência seria um qui-quadrado de 8 graus de liberdade, média 8; aqui é só uma estatística monótona de evidência conjunta). (3) CUSUM: C_t = max(0, C_(t-1) + F_t - kappa), com kappa = 8 (a média nula), zerado nos instantes de `reset` de produção (máscara fechada ou partida). (4) Alarme: C > h por >= 2 h, refratário de 72 h (a política fixa de `nv.politica`). O limiar h é calibrado por bisseção (monótona) para o TOTAL de horas de alarme igual ao do publicado (1.457 h em todo o período vigiado): SEM nenhuma janela de trip.
COMPARAÇÕES (todas ao mesmo custo total de 1.457 h, sem rótulos): M0 = o voto de produção (`ci.prever`, caixa-preta; 7/8, 9 FP); M1 = a fusão acima; M2 = a estatística de Fisher INSTANTÂNEA sem CUSUM (limiar por bisseção): isola o que o CUSUM acrescenta. Segunda leitura (para comparar com os EXP anteriores): M1 e M2 a 407 h fora das janelas (com rótulos).
SENSIBILIDADE de M1: W em {30, 60, 90} dias e kappa em {8, 10}.
PLACEBO: o alarme de cada método (custo total fixado sem rótulos) contra eventos deslocados de +45, +90, +135, +180 e +225 dias: a detecção ao acaso.
AUDITORIA DO FUTURO: embaralhar os sinais depois de 01/09/2025 e exigir que F e C até 31/08/2025 fiquem idênticos.
CRITÉRIOS (fixados antes). M1 GANHA do voto se, ao custo total igual, detecção >= 7/8 e FP <= 7 (FP -2), ou detecção 8/8 com FP <= 9; EQUIVALE se detecção >= 6/8 e FP <= 11; PIOR se o resto. Ruído com 8 eventos de +-2: um GANHO só vale se for visível nas sensibilidades (W e kappa) em >= 4 das 6 variantes.
EXPECTATIVAS: M1 alcança >= 6/8 ao custo total de 1.457 h ~45%; M1 tem mais FP que o voto (9) ~70% (o voto é ajustado, o Fisher não); M1 GANHA do voto ~8%; M2 (sem CUSUM) é pior que M1 em detecção ~70%; M1 estável nas sensibilidades (faixa de detecção <= 1) ~55%; a auditoria passa ~95%; o placebo de M1 fica <= 3/8 ~70%.
INTERPRETAÇÃO PREVISTA. Se M1 chega perto do voto sem nenhum rótulo e com dois parâmetros (W, kappa), o desenho coerente de fusão é uma alternativa menos sujeita a sobreajuste, candidata a hipótese prospectiva. Se fica muito abaixo, o voto de produção carrega informação (estrutura do portão mecânico e dois níveis) que a probabilidade de cauda combinada não captura. Se ganha, é o resultado que o usuário procura, a confirmar com placebo e o 13b.
LIMITES. 8 eventos; a combinação de Fisher ignora a dependência entre canais e a autocorrelação (não é um p-valor); a referência dos 60 dias anteriores contém alarmes; o custo total de 1.457 h inclui horas dentro das janelas; nada se adota sem sombra.

RESULTADO (16/10/2026) -- A FUSÃO PROBABILÍSTICA (Fisher + CUSUM) É MUITO PIOR QUE O VOTO DE PRODUÇÃO nas seis variantes; a auditoria do futuro passa.
  M0 (voto de produção, caixa-preta): 7/8, início 6, 9 FP (0,81/mês), 1.457 h totais, 407 h fora; placebo ~1,3/8. AUDITORIA DO FUTURO: F e C idênticos antes de 01/09/2025 com os sinais embaralhados depois (diferença 0): PASSA.
  M1 (Fisher + CUSUM), custo TOTAL de 1.457 h sem rótulos (detecção | FP | horas fora das janelas | placebo): W = 30 d: kappa 8: 1/8, 7 FP, 1.158 h, 0,0 | kappa 10: 2/8, 8 FP, 1.181 h, 0,0; W = 60 d: kappa 8: 2/8, 7 FP, 936 h, 1,3 | kappa 10: 2/8, 7 FP, 975 h, 1,3; W = 90 d: kappa 8: 2/8, 7 FP, 933 h, 1,3 | kappa 10: 4/8, 27 FP, 1.086 h, 3,2. M2 (Fisher instantâneo, sem CUSUM): W = 30: 3/8, 31 FP | W = 60: 6/8, início 4, 42 FP (3,77/mês), placebo 3,2 | W = 90: 3/8, 38 FP. Segunda leitura (407 h fora das janelas): M1 (W = 60, kappa = 8) 1/8, 5 FP.
  CLASSIFICAÇÃO: PIOR nas 6 variantes (GANHO em 0 e EQUIVALENTE em 0 das 6).
LEITURA. (1) Com as mesmas quatro entradas e o mesmo custo total, a fusão baseada em princípio detecta 1 a 4 de 8 contra 7/8 do voto de produção: o voto, o portão mecânico (nível B exige sp ou vb), os dois níveis, o CUSUM por canal com carga_reset e o refratário carregam informação que a probabilidade de cauda combinada não captura. (2) O mecanismo visível nas horas: 60 a 80% das horas de alarme do M1 estão FORA das janelas de trip (936 a 1.181 de 1.457 h), isto é, o orçamento de horas é consumido por poucos episódios longos (7 episódios de FP): o CUSUM sobre Fisher se acumula nos estados persistentes (sp e vb acesos 24 a 52% do tempo normal: 'canais são estados') e produz alarmes de dias; sem o CUSUM (M2) a detecção sobe a 6/8 mas com 31 a 42 episódios de FP (regime denso, placebo 2,3 a 3,2): o CUSUM trocou notificações por duração. (3) O resultado vale para ESTE desenho de fusão (Fisher, kappa na média nula, referência dos 60 dias anteriores com alarmes dentro, sem portão): não prova que nenhuma fusão baseada em princípio seja possível; mostra que a que parece natural é muito pior e que a estrutura do voto de produção não é arbitrária. (4) Resumo do que se testou contra o v2 com rigor crescente (calibração sem rótulos, placebo, auditoria do futuro): escores alternativos (EXP 20, 33, 34) e fusão probabilística (EXP 35) não o superam.
PLACAR: ACERTEI: a auditoria passa (~95%); o placebo de M1 <= 3/8 (~70%, exceto uma variante); nenhum GANHO (~92%). ERREI: M1 alcança >= 6/8 (~45%: no máximo 4/8); M1 com mais FP que o voto (~70%: tem menos FP, 7, com detecção 1-2); M2 pior que M1 em detecção (~70%: M2 é melhor, 6/8 em W = 60, a 42 FP); M1 estável nas sensibilidades (~55%: 1 a 4/8).

Uso:  PYTHONPATH=src .venv/bin/python experiments/35_fusao_probabilistica.py
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
E34 = _carrega("exp34", "34_subespacos_sem_vazamento.py"); E20 = E34.E20
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2
from nv.politica import episodios, alarme_de, calibra_limiar

ROOT = E34.ROOT; INICIO8 = E34.INICIO8; CANAIS = ("t", "p", "sp", "vb"); TOTAL_PUB = 1457.0


def caudas(proc, W):
    """u_c(t): probabilidade de cauda empírica contra os instantes vigiados dos W dias anteriores ao dia 1 do mês (causal)."""
    mask = proc["mask"].astype(bool).to_numpy(); idx = proc.index; U = {c: np.full(len(proc), np.nan) for c in CANAIS}
    for m0 in pd.date_range(INICIO8.replace(day=1), idx[-1], freq="MS", tz="UTC"):
        m1 = m0 + pd.offsets.MonthBegin(1); a = idx.searchsorted(m0 - pd.Timedelta(days=W)); b = idx.searchsorted(m0); i0 = idx.searchsorted(m0); i1 = idx.searchsorted(m1)
        for c in CANAIS:
            v = proc[c].to_numpy(); ref = np.sort(v[a:b][mask[a:b] & np.isfinite(v[a:b])])
            if len(ref) < 2000:
                continue
            x = v[i0:i1]; ok = np.isfinite(x) & mask[i0:i1]; cnt = len(ref) - np.searchsorted(ref, x[ok], side="left"); u = np.full(len(x), np.nan); u[ok] = (cnt + 1) / (len(ref) + 1); U[c][i0:i1] = u
    return U


def fisher(U):
    L = np.column_stack([np.log(U[c]) for c in CANAIS]); return -2.0 * L.sum(axis=1)


def cusum(F, reset, kappa):
    C = np.zeros(len(F)); c = 0.0
    for i in range(len(F)):
        if reset[i] or not np.isfinite(F[i]):
            c = 0.0
        else:
            c = max(0.0, c + F[i] - kappa)
        C[i] = c
    return C


def bis(serie, vig, reg, metrica, alvo, n_iter=28):
    e = serie.dropna(); lo, hi = float(np.nanquantile(e, 0.50)), float(e.max()); f = lambda t: metrica(avalia(alarme_de(episodios(serie, t), serie.index), vig, reg, inicio_periodo=INICIO8))
    for _ in range(n_iter):
        mid = float(np.sqrt(max(lo, 1e-12) * hi))
        if f(mid) > alvo:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(max(lo, 1e-12) * hi))


def placebo(al, vig, reg):
    out = []
    for d in (45, 90, 135, 180, 225):
        sh = [t + pd.Timedelta(days=d) for t in reg if t + pd.Timedelta(days=d) <= al.index[-1]]
        if sh:
            r = avalia(al, vig, sh, inicio_periodo=INICIO8); out.append(r.det / r.n_eventos)
    return float(np.median(out)) * 8


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index
    vig = vig_c.reindex(idx).fillna(False); reset = proc["reset"].to_numpy(bool)
    lin = lambda r: f"{r.det}/8, início {r.inicio}, FP {r.fp} ({r.fp_mes:.2f}/mês), totais {r.horas_alarme:.0f} h, fora {r.horas_alarme_fora:.0f} h"
    res0 = decide_v2(proc, det); al0 = res0["is_anomaly"].astype(bool); r0 = avalia(al0, vig, reg, inicio_periodo=INICIO8)
    print(f"M0 voto de produção: {lin(r0)} | placebo ~{placebo(al0, vig, reg):.1f}/8", flush=True)
    print("\nAUDITORIA DO FUTURO (sinais embaralhados depois de 01/09/2025)", flush=True)
    corte = pd.Timestamp("2025-09-01", tz="UTC"); p2 = proc.copy(); pos = idx >= corte; p2.loc[pos, CANAIS] = p2.loc[pos, CANAIS].sample(frac=1.0, random_state=1).to_numpy()
    Fa = fisher(caudas(proc, 60)); Fb = fisher(caudas(p2, 60)); sl = (idx >= INICIO8) & (idx < corte); d = float(np.nanmax(np.abs(Fa[sl] - Fb[sl]))); Ca = cusum(Fa, reset, 8); Cb = cusum(Fb, reset, 8); d2 = float(np.nanmax(np.abs(Ca[sl] - Cb[sl])))
    print(f"   F: diferença máxima {d:.2e}; C: {d2:.2e} -> {'PASSA' if d < 1e-9 and d2 < 1e-9 else 'FALHA'}", flush=True)
    print("\nM1 (Fisher + CUSUM) e M2 (Fisher instantâneo), custo TOTAL de 1.457 h sem rótulos; entre parênteses o placebo", flush=True); res = {}
    for W in (30, 60, 90):
        F = fisher(caudas(proc, W)); Fs = pd.Series(F, index=idx)
        for kappa in (8, 10):
            C = pd.Series(cusum(F, reset, kappa), index=idx); h = bis(C, vig, reg, lambda r: r.horas_alarme, TOTAL_PUB); al = alarme_de(episodios(C, h), idx); r = avalia(al, vig, reg, inicio_periodo=INICIO8); res[(W, kappa)] = r
            print(f"   M1 W = {W:2d} d, kappa = {kappa:2d}: {lin(r)} | placebo ~{placebo(al, vig, reg):.1f}/8", flush=True)
        h2 = bis(Fs, vig, reg, lambda r: r.horas_alarme, TOTAL_PUB); al2 = alarme_de(episodios(Fs, h2), idx); r2 = avalia(al2, vig, reg, inicio_periodo=INICIO8)
        print(f"   M2 W = {W:2d} d (sem CUSUM): {lin(r2)} | placebo ~{placebo(al2, vig, reg):.1f}/8", flush=True)
    print("\nSEGUNDA LEITURA (407 h fora das janelas, com rótulos; só para comparar com os EXP anteriores)", flush=True)
    F = fisher(caudas(proc, 60)); C = pd.Series(cusum(F, reset, 8), index=idx); h = calibra_limiar(C, vig, reg, 407.0, inicio=INICIO8); r = avalia(alarme_de(episodios(C, h), idx), vig, reg, inicio_periodo=INICIO8)
    print(f"   M1 (W = 60, kappa = 8): {lin(r)}", flush=True)
    print("\nCLASSIFICAÇÃO de M1 contra o voto de produção (custo total igual)", flush=True); cls = {}
    for k, r in res.items():
        cls[k] = "GANHO" if (r.det >= 7 and r.fp <= 7) or (r.det >= 8 and r.fp <= 9) else "EQUIVALENTE" if (r.det >= 6 and r.fp <= 11) else "PIOR"
        print(f"   W = {k[0]:2d}, kappa = {k[1]:2d}: {r.det}/8, FP {r.fp} -> {cls[k]}", flush=True)
    print(f"   GANHO em {sum(v == 'GANHO' for v in cls.values())} das 6 variantes (limite escrito: >= 4); EQUIVALENTE ou melhor em {sum(v != 'PIOR' for v in cls.values())} das 6", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXP 41 — CALIBRAÇÃO POR QUANTIL DO NORMAL RECENTE ('conforme') E CONTROLE ADAPTATIVO DO ORÇAMENTO DE ALARME, no lugar dos 8 limiares ajustados à mão.

PRÉ-REGISTRO (16/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. O ponto publicado tem 8 limiares de nível (k de A e B, 4 canais) ajustados nos mesmos eventos e é o topo de um pico: perturbando-os em +-10%, só 10% dos sorteios mantêm o desempenho (EXP 24, 30). Uma calibração em PROBABILIDADE (a idéia conforme: o limiar de um canal é um quantil da distribuição recente do próprio canal em operação normal) tem MENOS parâmetros e nenhum rótulo, e um controle adaptativo do orçamento de horas de alarme mantém a carga mensal estável sob deriva. A hipótese é de ROBUSTEZ e de estabilidade, não de ganho de detecção.
O QUE SE FAZ (8 eventos; os sinais de produção t, p, sp, vb e a decisão `ci.prever` intocada, só com os sinais reescalados e os limiares mudados):
  NORMALIZAÇÃO CAUSAL: cada sinal é dividido, mês a mês, pelo percentil pi0 = 0,95 dos instantes vigiados dos W = 60 dias ANTERIORES ao dia 1 do mês (sem rótulo; o W e o pi0 são variados abaixo).
  M1 ESTÁTICO: base = 1 em todos os canais, k_nivel_a = kappa_A igual nos 4 canais e k_nivel_b = 1,5 x kappa_A (1,5 = a média das razões k_B/k_A do ponto publicado: 1,55; 1,42; 1,9; 1,1); os demais botões de produção iguais. kappa_A é escolhido pela grade {1,0; 1,2; ..., 8,0} (e uma grade fina) como o que dá o TOTAL de horas de alarme igual ao do publicado (1.457 h) em todo o período vigiado: SEM janela de trip.
  M2 ADAPTATIVO: kappa_m mensal; kappa_(m+1) = kappa_m x ((horas realizadas em m + 10)/(orçamento de m + 10))^0,5, limitado a [0,5, 2] x kappa de M1; o orçamento do mês é 1.457 h x (horas vigiadas do mês / horas vigiadas totais). Controle de realimentação sobre o próprio alarme, sem rótulos.
  M0: o voto de produção como está (7/8, 9 FP, 1.457 h).
MEDIDAS: detecção de 8, início, FP, FP/mês, horas totais, e a DISPERSÃO MENSAL das horas de alarme (coeficiente de variação entre os meses vigiados e o máximo mensal); placebo (eventos deslocados +45 a +225 dias no mesmo alarme); ROBUSTEZ: 100 sorteios de kappa_A x U(0,9; 1,1) e razão x U(0,9; 1,1) (e 100 em +-20%) em torno de M1: fração 'BOA' = detecção >= a do centro - 1, FP <= o do centro + 2 e horas <= 1,3 x as do centro (o equivalente do que o EXP 24 usou para o publicado: 10% em +-10%). SENSIBILIDADE de M1: W em {30, 60, 90} dias e pi0 em {0,90; 0,95; 0,99}. AUDITORIA DO FUTURO: com os sinais embaralhados depois de 01/09/2025 os sinais normalizados até 31/08/2025 ficam idênticos.
CRITÉRIOS (fixados antes). M1 EQUIVALE ao voto se detecção >= 6/8 e FP <= 11 ao custo total igual; GANHA se detecção >= 7/8 e FP <= 7. A ROBUSTEZ DE M1 GANHA se a fração BOA em +-10% é >= 60% (contra 10% do publicado) e em +-20% >= 30%. M2 ESTABILIZA as horas se o coeficiente de variação mensal é >= 30% menor que o do voto e a detecção >= 6/8. Só vale um resultado visível em >= 4 das 9 variantes de sensibilidade.
EXPECTATIVAS: M1 EQUIVALE ao voto no centro ~45%, GANHA ~6%; a robustez de M1 em +-10% >= 60% ~35% (menos parâmetros, mas a decisão continua em degraus); M2 estabiliza as horas mensais ~55%; M1 com detecção >= 6/8 em >= 4 das 9 variantes ~40%.
INTERPRETAÇÃO PREVISTA. Se M1 empata em detecção e tem robustez >= 60%, é um ponto de operação MAIS HONESTO que o publicado (a mesma detecção típica, sem o pico): hipótese prospectiva no sombra.py. Se M2 estabiliza a carga mensal, é o argumento operacional para controle adaptativo do orçamento. Se M1 fica muito abaixo (<= 5/8), os limiares por canal ajustados à mão do v2 carregam informação (a combinação dos 4 canais com pesos diferentes) que a normalização uniforme perde.
LIMITES. 8 eventos; a razão 1,5 e a política de produção de refratário/escalada/duração permanecem as do v2; o p95 dos 60 dias anteriores contém alarmes; o custo total de 1.457 h inclui horas dentro das janelas; nada se adota sem sombra.

DESVIO DO PRÉ-REGISTRO (registrado depois da 1.ª execução): a grade de kappa_A começava em 1,0 e em 6 de 9 variantes o orçamento de 1.457 h não era alcançável nesse piso (horas 696 a 1.215 com kappa = 1,0), então o custo igual não foi atingido; a grade foi estendida para baixo (0,3 a 8,0, 17 valores) e um erro de código na etapa de robustez foi corrigido. Nada mais mudou.

RESULTADO (16/10/2026) -- A CALIBRAÇÃO POR QUANTIL COM DOIS PARÂMETROS PERDE MUITA DETECÇÃO: a robustez medida é a de um detector ruim; o controle adaptativo não estabiliza a carga mensal.
  M0 (voto de produção): 7/8, início 6, 9 FP (0,81/mês), 1.457 h; horas por mês: média 97, coeficiente de variação 1,68, máximo 614; placebo ~1,3/8. AUDITORIA DO FUTURO: PASSA (diferença 0).
  M1 ESTÁTICO (kappa_A pelo total de ~1.457 h sem rótulos, depois de estender a grade para baixo; razão 1,5): detecção/FP/horas por variante (W, pi0): (30, 0,90) 0/8, 10, 1.077 h; (30, 0,95) 1/8, 13, 1.003 h; (30, 0,99) 1/8, 16, 1.310 h; (60, 0,90) 5/8, 12, 1.565 h; (60, 0,95) 2/8, 12, 1.573 h; (60, 0,99) 0/8, 11, 1.410 h; (90, 0,90) 4/8, 14, 1.333 h; (90, 0,95) 2/8, 11, 1.334 h; (90, 0,99) 0/8, 10, 1.339 h. O centro (60, 0,95, kappa_A = 0,8): 2/8, início 1, 12 FP (1,08/mês), 1.573 h, CV mensal 1,61, placebo ~1,6/8. A detecção é de 0 a 5 de 8 nas 9 variantes (nenhuma >= 6/8); M1 NÃO EQUIVALE ao voto (7/8, 9 FP) e não ganha.
  M2 ADAPTATIVO: 4/8, início 2, 12 FP (1,08/mês), 1.866 h; horas por mês: média 124, CV 1,50 (11% menor que o 1,68 do voto, abaixo dos 30% exigidos), máximo 616; o kappa mensal bate no piso de 0,4 em 9 de 15 meses.
  ROBUSTEZ DE M1: +-10%: fração BOA 100% (detecção mediana 2/8, mín 1, máx 4; FP mediano 12); +-20%: 96% (detecção mediana 3/8; FP mediano 12, de 9 a 18). Formalmente passa (>= 60% e >= 30%), mas é a robustez de um detector que detecta 2 a 3 de 8: não vale como ganho (o publicado: 10% e 2%, mas com 7/8).
  CRITÉRIOS: M1 EQUIVALE: NÃO; M1 GANHA: NÃO; robustez de M1: passa formalmente, sem valor (detecção 2/8); M2 estabiliza as horas: NÃO; M1 com detecção >= 6/8 em >= 4 das 9 variantes: NÃO (0 de 9, máximo 5).
LEITURA. (1) Os limiares por canal do v2 (k de A e B por canal: t 1,1/1,7; p 1,2/1,7; sp 0,9/1,7; vb 2,0/2,2) carregam informação que a normalização uniforme por quantil perde: com uma única escala e o mesmo orçamento total de horas, a detecção cai de 7/8 para 2/8 no centro. Os canais do v2 são ESTADOS que vivem acesos em frações muito diferentes (24 a 52% do tempo normal, memória 'o-detector-esta-numa-fronteira'): dar a todos a mesma probabilidade de cauda (p95) é a calibração errada para um sistema em que o vb exige 2,0 e o sp só 0,9. (2) A tendência por pi0 (0,90 > 0,95 > 0,99: 5/8, 2/8, 0/8 em W = 60) mostra que os limiares do v2 correspondem a caudas muito mais altas em probabilidade do que o p95: calibrar 'em probabilidade' exigiria uma probabilidade por canal, isto é, voltar aos 8 parâmetros. (3) A idéia de controle adaptativo do orçamento de horas não estabilizou a carga mensal (CV 1,50 contra 1,68) e aumentou as horas totais (1.866 h): a carga mensal do v2 vem de poucos episódios longos (EXP 26/36), que um controlador mensal por realimentação não alcança. (4) Fecha-se a idéia de calibração conforme com estes dados; o ponto de operação publicado continua sendo o topo de um pico, sem alternativa de calibração com menos parâmetros que preserve a detecção.
PLACAR: ERREI: M1 EQUIVALE no centro (~45%: 2/8); M2 estabiliza as horas mensais (~55%: CV 1,50 contra 1,68); M1 com detecção >= 6/8 em >= 4 das 9 variantes (~40%: 0). ACERTEI: a robustez de M1 >= 60% em +-10% (~35%: 100%, mas de um detector ruim); GANHO de M1 (~6%: não).

Uso:  PYTHONPATH=src .venv/bin/python experiments/41_calibracao_conforme.py
"""
from __future__ import annotations
import contextlib, copy, io, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); CANAIS = ("t", "p", "sp", "vb"); TOTAL = 1457.0; RAZAO = 1.5; GL = {}


def normaliza(proc, W=60, pi0=0.95):
    idx = proc.index; mask = proc["mask"].to_numpy(bool); out = proc.copy(); Qm = {}
    for c in CANAIS:
        v = proc[c].to_numpy(); x = np.full(len(idx), np.nan)
        for m0 in pd.date_range(INICIO8.replace(day=1), idx[-1], freq="MS", tz="UTC"):
            m1 = m0 + pd.offsets.MonthBegin(1); a = idx.searchsorted(m0 - pd.Timedelta(days=W)); b = idx.searchsorted(m0); r = v[a:b][mask[a:b] & np.isfinite(v[a:b])]
            if len(r) >= 2000:
                q = float(np.quantile(r, pi0)); i0, i1 = idx.searchsorted(m0), idx.searchsorted(m1); x[i0:i1] = v[i0:i1] / q; Qm[(c, m0)] = q
        out[c] = x
    return out


def roda(pn, kA, razao=RAZAO, mensal=None):
    """Decisão de produção sobre os sinais normalizados com base 1; `mensal` = {Timestamp do mês: kappa_m} (multiplica o limiar por mês)."""
    p2 = pn.copy()
    if mensal is not None:
        f = np.ones(len(p2))
        for m0, k in mensal.items():
            i0, i1 = p2.index.searchsorted(m0), p2.index.searchsorted(m0 + pd.offsets.MonthBegin(1)); f[i0:i1] = k
        for c in CANAIS:
            p2[c] = p2[c] / f
        kA = 1.0
    d = copy.deepcopy(GL["det"]); d["base"] = {c: 1.0 for c in CANAIS}; d["k_nivel_a"] = {c: kA for c in CANAIS}; d["k_nivel_b"] = {c: razao * kA for c in CANAIS}
    with contextlib.redirect_stdout(io.StringIO()):
        R = GL["ci"].prever({"detector": d}, p2)
    return R["is_anomaly"].astype(bool)


def avalia_al(al):
    return avalia(al.reindex(GL["al_c"].index).fillna(False), GL["vig"], GL["reg"], inicio_periodo=INICIO8)


def tarefa(a):
    chave, kA = a; r = avalia_al(roda(GL["pn"][chave[0]], kA, razao=chave[1])); return (chave, kA), (r.det, r.fp, r.horas_alarme, r.horas_alarme_fora)


def mensais(al, vig, idx):
    s = al.reindex(vig.index).fillna(False) & vig; s = s.loc[INICIO8:]; return (s.groupby([s.index.year, s.index.month]).sum() * 2 / 60)


def placebo(al, reg):
    out = []
    for d in (45, 90, 135, 180, 225):
        sh = [t + pd.Timedelta(days=d) for t in reg if t + pd.Timedelta(days=d) <= al.index[-1]]
        if sh:
            r = avalia(al.reindex(GL["al_c"].index).fillna(False), GL["vig"], sh, inicio_periodo=INICIO8); out.append(r.det / r.n_eventos)
    return float(np.median(out)) * 8


def main():
    ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        det = ci.carregar_modelos(PKG / "modelos")[-1]["detector"]
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index; vig = vig_c.reindex(al_c.index).fillna(False)
    GL.update(ci=ci, det=det, reg=reg, al_c=al_c, vig=vig)
    with contextlib.redirect_stdout(io.StringIO()):
        R0 = ci.prever({"detector": det}, proc)
    al0 = R0["is_anomaly"].astype(bool); r0 = avalia_al(al0); m0 = mensais(al0, vig_c.reindex(idx).fillna(False), idx); lin = lambda r: f"{r.det}/8, início {r.inicio}, FP {r.fp} ({r.fp_mes:.2f}/mês), totais {r.horas_alarme:.0f} h"
    cv = lambda m: float(m.std() / max(m.mean(), 1e-9))
    print(f"M0 voto de produção: {lin(r0)} | horas por mês: média {m0.mean():.0f}, CV {cv(m0):.2f}, máx {m0.max():.0f} | placebo ~{placebo(al0, reg):.1f}/8", flush=True)
    print("\nAUDITORIA DO FUTURO", flush=True); corte = pd.Timestamp("2025-09-01", tz="UTC"); p2 = proc.copy(); pos = idx >= corte; p2.loc[pos, list(CANAIS)] = p2.loc[pos, list(CANAIS)].sample(frac=1.0, random_state=1).to_numpy()
    na, nb = normaliza(proc), normaliza(p2); sl = (idx >= INICIO8) & (idx < corte); d = max(float(np.nanmax(np.abs(na[c].to_numpy()[sl] - nb[c].to_numpy()[sl]))) for c in CANAIS); print(f"   diferença máxima dos sinais normalizados antes de 01/09/2025: {d:.2e} -> {'PASSA' if d < 1e-9 else 'FALHA'}", flush=True)
    variantes = [(W, pi0) for W in (30, 60, 90) for pi0 in (0.90, 0.95, 0.99)]; GL["pn"] = {(W, pi0): normaliza(proc, W, pi0) for (W, pi0) in variantes}; grade = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]
    with Pool(4) as p:
        res = dict(p.map(tarefa, [((v, RAZAO), k) for v in variantes for k in grade], chunksize=4))
    print("\nM1 ESTÁTICO (kappa_A pelo TOTAL de 1.457 h, sem rótulos; razão k_B/k_A = 1,5): detecção | FP | horas totais | kappa_A", flush=True); centros = {}
    for v in variantes:
        k = min(grade, key=lambda k: abs(res[((v, RAZAO), k)][2] - TOTAL)); r = res[((v, RAZAO), k)]; centros[v] = (k, r)
        print(f"   W = {v[0]:2d} d, pi0 = {v[1]:.2f}: {r[0]}/8, FP {r[1]}, {r[2]:.0f} h (kappa_A = {k})", flush=True)
    v0 = (60, 0.95); kA0, rc = centros[v0]; al1 = roda(GL["pn"][v0], kA0); r1 = avalia_al(al1); m1 = mensais(al1, vig_c.reindex(idx).fillna(False), idx); print(f"\nM1 (W = 60, pi0 = 0,95, kappa_A = {kA0}): {lin(r1)} | horas por mês: média {m1.mean():.0f}, CV {cv(m1):.2f}, máx {m1.max():.0f} | placebo ~{placebo(al1, reg):.1f}/8", flush=True)
    meses = list(pd.date_range(INICIO8.replace(day=1), idx[-1], freq="MS", tz="UTC")); vh = (vig_c.reindex(idx).fillna(False) & (idx >= INICIO8)); vhm = {m: float(vh.loc[m:m + pd.offsets.MonthBegin(1) - pd.Timedelta("2min")].sum() * 2 / 60) for m in meses}; tot = sum(vhm.values()); kap = {}; k = kA0; kap[meses[0]] = k; pn = GL["pn"][v0]
    for i, m in enumerate(meses[:-1]):
        al = roda(pn, kA0, mensal=dict(kap)); s = (al.reindex(vig_c.index).fillna(False) & vig_c).loc[m:m + pd.offsets.MonthBegin(1) - pd.Timedelta("2min")]; real = float(s.sum() * 2 / 60); alvo = TOTAL * vhm[m] / tot
        k = float(np.clip(k * ((real + 10) / (alvo + 10)) ** 0.5, 0.5 * kA0, 2.0 * kA0)); kap[meses[i + 1]] = k
    al2 = roda(pn, kA0, mensal=kap); r2 = avalia_al(al2); m2 = mensais(al2, vig_c.reindex(idx).fillna(False), idx); print(f"M2 ADAPTATIVO: {lin(r2)} | horas por mês: média {m2.mean():.0f}, CV {cv(m2):.2f}, máx {m2.max():.0f} | placebo ~{placebo(al2, reg):.1f}/8 | kappa por mês: {[round(v, 2) for v in kap.values()]}", flush=True)
    print("\nROBUSTEZ DE M1 em torno do centro (kappa_A e razão perturbados juntos)", flush=True); rng = np.random.default_rng(20261016)
    for amp in (0.10, 0.20):
        ped = [(((60, 0.95), float(RAZAO * rng.uniform(1 - amp, 1 + amp))), float(kA0 * rng.uniform(1 - amp, 1 + amp))) for _ in range(100)]
        with Pool(4) as p:
            rs = p.map(tarefa2, ped, chunksize=4)
        boa = [(x[0] >= rc[0] - 1 and x[1] <= rc[1] + 2 and x[2] <= 1.3 * rc[2]) for x in rs]; d = [x[0] for x in rs]; f = [x[1] for x in rs]
        print(f"   ±{int(amp * 100)}%: fração BOA {100 * np.mean(boa):.0f}% | detecção mediana {np.median(d):.0f}/8 (mín {min(d)}, máx {max(d)}) | FP mediano {np.median(f):.0f} (mín {min(f)}, máx {max(f)}) | (publicado nos EXP 24/30: BOA 10% em ±10% e 2% em ±20%)", flush=True)


def tarefa2(a):
    (chave, razao), kA = a; r = avalia_al(roda(GL["pn"][chave], kA, razao=razao)); return (r.det, r.fp, r.horas_alarme)


if __name__ == "__main__":
    main()

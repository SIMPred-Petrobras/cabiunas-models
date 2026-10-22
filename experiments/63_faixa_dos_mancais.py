#!/usr/bin/env python3
"""EXP 63 — FAIXA FÍSICA PRÓPRIA PARA AS RTD DE MANCAL: o artefato de ~871 °C da máquina parada deixa de entrar no detector.

PRÉ-REGISTRO (21/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp63.py`), branch exp/modos-de-falha.

POR QUÊ (medido em 21/10, `manutencao-e-artefato-de-parada-21out`). Com a máquina PARADA o TI_0305 lê até 871 °C (1.474 amostras de 2 min > 200 °C, todas parada; TI_0303 27,
TI_0307 3; zero com a máquina ligada). A `_faixa` do pacote dá (−15, 900) °C a todo "_TI_" — larga por causa do T7 de exaustão (TI_0315/0317) — e o artefato passa como medição: o
escore cru do t nesses instantes é ~376x o p99 do treino. Como o EWMA roda sobre o sinal inteiro (parada incluída), na abertura da máscara o EWMA do t fica acima do limiar A em
24 de 48 aberturas com o artefato contra 21 sem. O spread do sp (TI_0305 contra a mediana dos irmãos) também recebe o artefato. É defeito de DADO, não escolha de modelo: o mancal
nunca passou de ~140 °C com a máquina ligada.

CORREÇÃO (fixada antes): `_faixa` devolve (0, 200) °C para TI_0301, TI_0303, TI_0305 e TI_0307 (RTD de mancal do compressor); fora disso vira NaN, como qualquer sentinela. As demais
tags não mudam. Aplicada por substituição de `ci._faixa` em memória (o pacote não é editado).
BRAÇO A (principal): replay causal semanal de produção (`sinais_causais`, mesmo início e complemento de 30 s do EXP 29/54) com a faixa nova -> `ci.prever` como caixa-preta.
BRAÇO B (robustez, só o canal t): harness fiel (`walk_forward(com_sk(PCAfiel))`) sobre a grade com a faixa nova, n = 20.000 e 19.000, contra o MESMO harness sem a correção
(`escores_sub34`); p, sp e vb do cache.
MEDIDAS: detecção de 8, início, banda, FP, FP/mês, horas totais e fora, lista dos episódios que mudam (nascimento, duração, TP/FP), EWMA do t e do sp na abertura da máscara.
CRITÉRIOS (fixados antes):
  HIGIENE OK = A: detecção 7/8, início >= 6, banda >= 5, FP <= 9, horas <= 1,05 x 1.457; E B: nos dois n, detecção não menor e FP não maior que o mesmo harness sem a correção.
  GANHO = HIGIENE OK e FP <= 8 em A. PIOR = A com detecção < 7 ou FP > 9. EQUIVALENTE no resto.
EXPECTATIVAS: V1-V3 passam (~90%); HIGIENE OK (~70%); GANHO (~30%); PIOR (~15%); no máximo 3 episódios mudam em A (~70%); o EWMA do t na abertura cai em >= 3 das 48 aberturas (~75%).
INTERPRETAÇÃO PREVISTA: HIGIENE OK -> entra na v2.1 como correção de dado (vale mesmo sem ganho: o modelo não deve ver 871 °C num mancal), validada na sombra. PIOR -> o artefato
estava, por acaso, ajudando alguma detecção; documentar qual e não corrigir às cegas.
LIMITES: 8 eventos; a faixa 0-200 °C é escolha de engenharia (setpoint de trip 125 °C, máximo ligado ~140 °C), não ajustada; o braço B só mede o t.
VERIFICAÇÕES: V1 `ci.prever` sobre o cache do controle dá 7/8 e FP 9; V2 no braço A, vb e p são idênticos ao cache em todo instante, e t e sp idênticos nos instantes vigiados sem
artefato nas 72 h anteriores (diferença <= 1e-6); V3 o harness sem a correção reproduz `escores_sub34` (diferença <= 1e-6).

Uso local (encanamento, janela curta, sem tabelas):  EXP63_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/63_faixa_dos_mancais.py
No worker:  PYTHONPATH=src python scripts/worker_exp63.py --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py")
from nv import io as nvio
from nv.avalia import avalia, carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.plugin import decide_v2, substitui
from nv.walk import walk_forward

TESTE = bool(os.environ.get("EXP63_TESTE"))
ROOT = HERE.parent / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); P = "954005_624_"
MANCAIS = (P + "TI_0301", P + "TI_0303", P + "TI_0305", P + "TI_0307"); FAIXA_MANCAL = (0.0, 200.0)
INICIO_REPLAY = pd.Timestamp("2025-01-08 07:34", tz="UTC") + pd.Timedelta(weeks=42 if TESTE else 0)   # no teste, um corte do calendário semanal do replay completo (29/10/2025: pega os artefatos de nov-dez)
CACHE = ROOT / ("sinais_63_teste.parquet" if TESTE else "sinais_63_faixa_mancal.parquet"); NS = (20000,) if TESTE else (20000, 19000); GL = {}


def faixa_nova(orig):
    def f(tag):
        return FAIXA_MANCAL if tag in MANCAIS else orig(tag)
    return f


def mede(al):
    a = al.reindex(GL["al_c"].index).fillna(False)
    r = avalia(a, GL["vig_c"], GL["trips"], inicio_periodo=INICIO8)
    return dict(det=r.det, inicio=r.inicio, banda=r.banda, fp=r.fp, fp_mes=r.fp_mes, horas=r.horas_alarme, fora=r.horas_alarme_fora), a


def pontua(a):
    nome, n, limpa = a
    g = GL["g"].copy()
    if limpa:
        for c in MANCAIS:
            g[c] = g[c].where((g[c] >= FAIXA_MANCAL[0]) & (g[c] <= FAIXA_MANCAL[1]))
    s = walk_forward(lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente), g, GL["tags_t"], GL["est"], INICIO8, GL["fim_walk"], n_treino=n, semente=0)
    return nome, n, s.loc[INICIO8 - pd.Timedelta(days=10):]


def main():
    t0 = time.time(); ci = _ci(); ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    proc0 = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc0["mask"] = proc0["mask"].astype(bool)
    GL["trips"] = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    al_c, vig_c = carrega_controle("atual"); GL.update(al_c=al_c, vig_c=vig_c)
    with contextlib.redirect_stdout(io.StringIO()):
        res0 = decide_v2(proc0, det)
    m0, a0 = mede(res0["is_anomaly"].astype(bool)); ok["V1"] = (m0["det"], m0["fp"]) == (7, 9)
    print(f"V1 — controle do cache: {m0['det']}/8, FP {m0['fp']} -> {'PASSA' if ok['V1'] else 'FALHA'}", flush=True)

    # ── braço A: replay causal com a faixa nova ──
    x = pd.read_parquet(ROOT / "raw30s_2024q4_2025q1.parquet")
    orig = ci._faixa; ci._faixa = faixa_nova(orig)
    try:
        SA = sinais_causais(INICIO_REPLAY, cache=CACHE, extra=x)
    finally:
        ci._faixa = orig
    procA = SA[["t", "p", "sp", "vb", "mask", "reset"]].copy(); procA["mask"] = procA["mask"].astype(bool)
    print(f"replay A: {len(procA)} instantes ({time.time() - t0:.0f} s)", flush=True)
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"]
    art = (g[list(MANCAIS)] > FAIXA_MANCAL[1]).any(axis=1) | (g[list(MANCAIS)] < FAIXA_MANCAL[0]).any(axis=1)
    longe = ~(art.astype(int).rolling(72 * 30, min_periods=1).max().astype(bool))       # sem artefato nas 72 h anteriores
    j = procA.index.intersection(proc0.index); j = j[j > INICIO_REPLAY]; vigj = proc0.loc[j, "mask"].astype(bool) & longe.reindex(j).fillna(False)
    dif = {c: float(np.nanmax(np.abs((procA.loc[j, c] - proc0.loc[j, c]).to_numpy()))) for c in ("vb", "p")}
    dif |= {c: float(np.nanmax(np.abs((procA.loc[j, c][vigj] - proc0.loc[j, c][vigj]).to_numpy()))) for c in ("t", "sp")}
    ok["V2"] = all(v <= 1e-6 for v in dif.values())
    mud = {c: (int((~np.isclose(procA.loc[j, c], proc0.loc[j, c], rtol=0, atol=1e-6, equal_nan=True)).sum()), int((~np.isclose(procA.loc[j, c][proc0.loc[j, "mask"].astype(bool)], proc0.loc[j, c][proc0.loc[j, "mask"].astype(bool)], rtol=0, atol=1e-6, equal_nan=True)).sum())) for c in ("t", "sp")}
    print(f"   instantes em que a correção muda o sinal (todos, vigiados): t {mud['t']}, sp {mud['sp']}", flush=True)
    print(f"V2 — A x cache: vb e p em todo instante; t e sp nos vigiados sem artefato nas 72 h anteriores: {({k: f'{v:.1e}' for k, v in dif.items()})} -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        resA = decide_v2(procA, det)
    mA, aA = mede(resA["is_anomaly"].astype(bool))
    thr = {c: det["base"][c] * det["k_nivel_a"][c] for c in ("t", "sp")}
    ab = proc0.index[np.flatnonzero(proc0["mask"].to_numpy()[1:] & ~proc0["mask"].to_numpy()[:-1]) + 1]; ab = ab[(ab >= INICIO8) & ab.isin(procA.index)]
    borda = {c: (int((proc0.loc[ab, c] > thr[c]).sum()), int((procA.loc[ab, c] > thr[c]).sum()), len(ab)) for c in ("t", "sp")}

    # ── braço B: harness fiel, só o t ──
    GL.update(g=g, est=est.to_numpy(bool), fim_walk=(INICIO8 + pd.Timedelta(days=20)) if TESTE else vig_c.index[-1], tags_t=list(modelos[-1]["normalizacao"]["temperatura"]["cols"]))
    with Pool(min(int(os.environ.get("EXP63_PROCS", 4)), 2 * len(NS))) as pool:
        SB = {(nome, n): s for nome, n, s in pool.imap_unordered(pontua, [(nome, n, limpa) for n in NS for nome, limpa in (("sem", False), ("com", True))], chunksize=1)}
    RB = {}
    for n in NS:
        ref = pd.read_parquet(ROOT / "escores_sub34" / f"t_A_{n}.parquet")["s"]; s0 = SB[("sem", n)]; jj = ref.index.intersection(s0.index)
        d3 = float(np.nanmax(np.abs(ref.loc[jj].to_numpy() - s0.loc[jj].to_numpy()))); ok[f"V3_{n}"] = d3 <= 1e-6
        print(f"V3 [n={n}] harness sem a correção x escores_sub34: {d3:.1e} -> {'PASSA' if ok[f'V3_{n}'] else 'FALHA'}", flush=True)
        for nome in ("sem", "com"):
            s = SB[(nome, n)]; e = s.ewm(halflife=pd.Timedelta(det["halflife"]["t"]), times=s.index).mean().where(s.notna())
            with contextlib.redirect_stdout(io.StringIO()):
                r = decide_v2(substitui(proc0, "t", e), det)
            RB[(nome, n)] = mede(r["is_anomaly"].astype(bool))[0]
    GL.update(m0=m0, a0=a0, mA=mA, aA=aA, borda=borda, RB=RB)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (janela curta; saída suprimida, nada gravado). {ok}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(ok, t0, escreve=True)


def relata(ok, t0, escreve):
    from nv.avalia.metricas import episodios_de_serie, JANELA
    m0, a0, mA, aA, borda, RB = GL["m0"], GL["a0"], GL["mA"], GL["aA"], GL["borda"], GL["RB"]
    loc = lambda t: (t - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    linha = lambda nome, m: print(f"   {nome:30s} {m['det']}/8 | início {m['inicio']} | banda {m['banda']} | FP {m['fp']:2d} | {m['fp_mes']:.3f}/mês | {m['horas']:5.0f} h | fora {m['fora']:4.0f} h", flush=True)
    print("\nBRAÇO A — replay causal de produção", flush=True)
    linha("controle (faixa −15..900)", m0); linha("faixa dos mancais 0..200", mA)
    print(f"   abertura da máscara com o canal acima do limiar A: t {borda['t'][0]} -> {borda['t'][1]} de {borda['t'][2]}; sp {borda['sp'][0]} -> {borda['sp'][1]} de {borda['sp'][2]}", flush=True)
    e0 = set(episodios_de_serie(a0.loc[INICIO8:])); eA = set(episodios_de_serie(aA.loc[INICIO8:]))
    tp = lambda a, b: "TP" if any(a <= t and b >= t - JANELA for t in GL["trips"]) else "FP"
    print("   episódios que SAEM: " + ("; ".join(f"{loc(a)} ({(b - a).total_seconds() / 3600:.0f} h, {tp(a, b)})" for a, b in sorted(e0 - eA)) or "nenhum"), flush=True)
    print("   episódios que ENTRAM: " + ("; ".join(f"{loc(a)} ({(b - a).total_seconds() / 3600:.0f} h, {tp(a, b)})" for a, b in sorted(eA - e0)) or "nenhum"), flush=True)
    print("\nBRAÇO B — harness fiel, só o t", flush=True)
    for (nome, n), m in sorted(RB.items(), key=lambda kv: (kv[0][1], kv[0][0] != "sem")):
        linha(f"n={n} {'sem' if nome == 'sem' else 'com'} a correção", m)
    NSs = sorted({n for _, n in RB})
    b_ok = all(RB[("com", n)]["det"] >= RB[("sem", n)]["det"] and RB[("com", n)]["fp"] <= RB[("sem", n)]["fp"] for n in NSs)
    a_ok = mA["det"] == 7 and mA["inicio"] >= 6 and mA["banda"] >= 5 and mA["fp"] <= 9 and mA["horas"] <= 1.05 * 1457
    cl = ("GANHO" if mA["fp"] <= 8 else "HIGIENE OK") if (a_ok and b_ok) else ("PIOR" if (mA["det"] < 7 or mA["fp"] > 9) else "EQUIVALENTE")
    print(f"\nCLASSIFICAÇÃO: braço A {'ok' if a_ok else 'não'} | braço B {'ok' if b_ok else 'não'} -> {cl}", flush=True)
    if escreve:
        pd.DataFrame([dict(braco="A controle", **m0), dict(braco="A faixa nova", **mA)] + [dict(braco=f"B n={n} {nome}", **m) for (nome, n), m in RB.items()]).to_csv(ROOT / "exp63_resultados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

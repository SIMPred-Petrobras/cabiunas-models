#!/usr/bin/env python3
"""EXP 62 — A REGRA "ALARME EM CARGA BAIXA VIRA ATENÇÃO": robustez a limiar, indicador de carga, janelas e placebo de permutação.

PRÉ-REGISTRO (21/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp62.py`), branch exp/modos-de-falha.

POR QUÊ. No EXP 61 a carga na notificação (T5 em percentil contra os 30 d anteriores) separou acerto de FP FORA da amostra (AUC 0,75 nos episódios novos; 0,79 no G168; 0,75 no
B800). O filtro "L < 0,5" — resultado SECUNDÁRIO, não critério — deu FP 4 a 6 com -0 a -1 detecção nas quatro bases, o que cumpre a meta do usuário. Antes de qualquer uso como
regra, é preciso saber se isso depende do limiar, do indicador, das janelas, e se é melhor que suprimir o mesmo número de episódios ao acaso.

REGRA: episódio (`ci._episodios`, desde 01/02/2025) com L < theta na notificação vira "atenção" (sai do alarme); o resto fica igual. O detector não muda.
CONFIGURAÇÃO PRINCIPAL (a do EXP 61): theta = 0,5; indicador T5_AVG_A; referência = 30 d anteriores de operação estável; janela de notificação = 2 h.
VIZINHANÇA (um fator por vez, 8 configurações): theta 0,4 e 0,6; indicador média dos TC382_01..06 (termopares de exaustão) e PDI_0317 (ΔP do gás combustível, ligado à vazão de
combustível); referência 15 d e 60 d; janela 1 h e 4 h.
PLACEBO DE PERMUTAÇÃO (configuração principal): em cada base, os valores de L embaralhados entre os episódios (500 vezes, semente 20261026) — suprime o MESMO número de episódios,
ao acaso. p_conj = fração das permutações com FP <= o da regra E detecção >= a da regra.
BASES: atual (já vista), G168, B800, atual n = 19.000 (as do EXP 60/61).
CRITÉRIOS (fixados antes):
  ROBUSTO = (a) configuração principal: FP <= 6 e perda <= 1 detecção nas 4 bases; E (b) em >= 6 das 8 configurações vizinhas, perda <= 1 detecção em toda base e FP médio das
  bases <= 7; E (c) p_conj < 0,10 em >= 3 das 4 bases.
  FRÁGIL = (a) vale mas (b) ou (c) não. NÃO REPLICA = (a) não vale.
EXPECTATIVAS: (a) vale (~95%, já visto); (b) vale (~50%); (c) vale (~55%); a média dos TC382 concorda com o T5 (~80%) e o PDI_0317 concorda (~45%); ROBUSTO (~35%).
INTERPRETAÇÃO PREVISTA: ROBUSTO -> candidata da v2.1 na camada de NOTIFICAÇÃO, na forma "atenção em carga baixa" (nunca suprimir: o operador continua vendo), a observar
prospectivamente nos ledgers da sombra (é pós-processamento; não toca no congelado). FRÁGIL -> só contexto de prioridade na tela. NÃO REPLICA -> idem, e o EXP 61 fica como
associação.
LIMITES: 8 eventos; bases compartilham dado; o T5 é indicador imperfeito de carga; o limiar 0,5 é a mediana (escolha natural, não ajustada); nada se adota sem a sombra.
VERIFICAÇÕES: V1 bases com as referências (atual 7/8 FP 9; G168 8/8 FP 14; B800 8/8 FP 12); V2 a configuração principal reproduz o secundário do EXP 61 (FP 4, 6, 6, 6);
V3 os três indicadores têm >= 30 amostras estáveis na referência de todo episódio.

Uso local (encanamento, sem tabelas):  EXP62_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/62_regra_de_carga.py
No worker:  PYTHONPATH=src python scripts/worker_exp62.py --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E60 = _carrega("exp60", "60_borda_adiada.py"); E61 = _carrega("exp61", "61_carga_na_notificacao.py")
from nv import io as nvio
from nv.avalia import carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG

TESTE = bool(os.environ.get("EXP62_TESTE"))
ROOT = HERE.parent / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); P = "954005_624_"
PRINC = dict(theta=0.5, ind="T5", ref_d=30, notif_h=2)
VIZ = [dict(PRINC, theta=0.4), dict(PRINC, theta=0.6), dict(PRINC, ind="TC382"), dict(PRINC, ind="PDI_0317"),
       dict(PRINC, ref_d=15), dict(PRINC, ref_d=60), dict(PRINC, notif_h=1), dict(PRINC, notif_h=4)]
N_PERM = 50 if TESTE else 500; SEMENTE = 20261026


def L_de(serie, a, ref_d, notif_h):
    w = serie.loc[a:a + pd.Timedelta(hours=notif_h)].dropna(); r = serie.loc[a - pd.Timedelta(days=ref_d):a - pd.Timedelta(minutes=2)].dropna()
    if len(w) == 0 or len(r) < 30:
        return np.nan, len(r)
    return float((r < float(w.median())).mean()), len(r)


def aplica(al, eps, Ls, theta):
    a = al.copy()
    for (ini, fim), L in zip(eps, Ls):
        if np.isfinite(L) and L < theta:
            a.loc[ini:fim] = False
    return a


def main():
    t0 = time.time(); ci = _ci(); E60.GL["ci"] = ci; ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    E60.GL["trips"] = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"]
    IND = {"T5": g["T5_AVG_A"].where(est), "TC382": g[[f"TC382_0{i}_A" for i in range(1, 7)]].mean(axis=1).where(est), "PDI_0317": g[P + "PDI_0317"].where(est)}
    B = {v: carrega_controle(v) for v in ("atual", "G168", "B800")}
    E60.GL["idx"] = B["atual"][0].index
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    B["atual n=19000"] = (E60.base_n(ci, det, proc, 19000).reindex(E60.GL["idx"]).fillna(False), B["atual"][1])
    for v, (dr, fr) in E60.REF.items():
        m = E60.mede(*B[v]); ok[f"V1_{v}"] = (m["det"], m["fp"]) == (dr, fr)
    EPS = {v: [(a, b) for a, b in ci._episodios(al.loc[INICIO8:])] for v, (al, vig) in B.items()}
    base0 = {v: E60.mede(*B[v]) for v in B}

    def avalia_cfg(cfg):
        out = {}; nref_min = 10 ** 9
        for v, (al, vig) in B.items():
            Ls = []
            for a, b in EPS[v]:
                L, nr = L_de(IND[cfg["ind"]], a, cfg["ref_d"], cfg["notif_h"]); Ls.append(L); nref_min = min(nref_min, nr)
            m = E60.mede(aplica(al, EPS[v], Ls, cfg["theta"]), vig); out[v] = (m, Ls)
        return out, nref_min
    R0, nmin = avalia_cfg(PRINC)
    fps = [R0[v][0]["fp"] for v in ("atual", "G168", "B800", "atual n=19000")]
    ok["V2"] = fps == [4, 6, 6, 6]
    print(f"V1 — bases com as referências: {all(ok[k] for k in ok if k.startswith('V1'))} | V2 — principal reproduz o EXP 61 (FP 4, 6, 6, 6): {fps} -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    RV = []; nmins = [nmin]
    for cfg in VIZ:
        r, nm = avalia_cfg(cfg); RV.append((cfg, r)); nmins.append(nm)
    ok["V3"] = min(nmins) >= 30
    print(f"V3 — referência com >= 30 amostras em todo episódio e indicador: mínimo {min(nmins)} -> {'PASSA' if ok['V3'] else 'FALHA'} ({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(SEMENTE); PERM = {}
    for v, (al, vig) in B.items():
        m_r = R0[v][0]; Ls = np.array(R0[v][1]); fin = np.isfinite(Ls); cont = 0
        for _ in range(N_PERM):
            Lp = Ls.copy(); Lp[fin] = rng.permutation(Ls[fin]); mp = E60.mede(aplica(al, EPS[v], Lp, PRINC["theta"]), vig)
            cont += (mp["fp"] <= m_r["fp"]) and (mp["det"] >= m_r["det"])
        PERM[v] = cont / N_PERM
    GL = dict(B=B, base0=base0, R0=R0, RV=RV, PERM=PERM)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(GL, ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (saída suprimida, nada gravado). {ok}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(GL, ok, t0, escreve=True)


def relata(GL, ok, t0, escreve):
    B, base0, R0, RV, PERM = GL["B"], GL["base0"], GL["R0"], GL["RV"], GL["PERM"]
    nome = lambda c: f"theta {c['theta']} | {c['ind']} | ref {c['ref_d']} d | notif {c['notif_h']} h"
    print("\nCONFIGURAÇÃO PRINCIPAL (" + nome(PRINC) + ")   det/8 | FP | horas  (original -> regra; perdidos)", flush=True)
    for v in B:
        m0 = base0[v]; m1 = R0[v][0]
        print(f"   {v:14s} {m0['det']}/8 FP {m0['fp']:2d} {m0['horas']:5.0f} h -> {m1['det']}/8 FP {m1['fp']:2d} {m1['horas']:5.0f} h | perdidos {[e for e in m0['ev'] if m0['ev'][e] and not m1['ev'][e]]} | placebo p_conj {PERM[v]:.3f}", flush=True)
    print("\nVIZINHANÇA (det/8 e FP por base: atual | G168 | B800 | n=19000)", flush=True)
    viz_ok = 0
    for cfg, r in RV:
        perdas = [base0[v]["det"] - r[v][0]["det"] for v in B]; fpm = float(np.mean([r[v][0]["fp"] for v in B])); bom = max(perdas) <= 1 and fpm <= 7
        viz_ok += bom
        print(f"   {nome(cfg):42s} | " + " | ".join(f"{r[v][0]['det']}/8 FP {r[v][0]['fp']:2d}" for v in B) + f" | FP médio {fpm:.1f} -> {'ok' if bom else 'não'}", flush=True)
    a = all(R0[v][0]["fp"] <= 6 and base0[v]["det"] - R0[v][0]["det"] <= 1 for v in B); b = viz_ok >= 6; c = sum(PERM[v] < 0.10 for v in B) >= 3
    cl = "ROBUSTO" if (a and b and c) else "FRÁGIL" if a else "NÃO REPLICA"
    print(f"\nCLASSIFICAÇÃO: (a) principal {'ok' if a else 'não'} | (b) vizinhança {viz_ok}/8 {'ok' if b else 'não'} | (c) placebo p < 0,10 em {sum(PERM[v] < 0.10 for v in B)}/4 bases {'ok' if c else 'não'} -> {cl}", flush=True)
    if escreve:
        linhas = [dict(cfg=nome(PRINC), base=v, det=R0[v][0]["det"], fp=R0[v][0]["fp"], horas=R0[v][0]["horas"], p_conj=PERM[v]) for v in B]
        linhas += [dict(cfg=nome(cfg), base=v, det=r[v][0]["det"], fp=r[v][0]["fp"], horas=r[v][0]["horas"]) for cfg, r in RV for v in B]
        pd.DataFrame(linhas).to_csv(ROOT / "exp62_resultados.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

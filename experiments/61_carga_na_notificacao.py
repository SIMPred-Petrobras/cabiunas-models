#!/usr/bin/env python3
"""EXP 61 — A CARGA NA NOTIFICAÇÃO SEPARA ACERTO DE FALSO POSITIVO? (teste fora da amostra do achado exploratório de 21/10)

PRÉ-REGISTRO (21/10/2026), COMMITADO ANTES DE RODAR. Roda no worker do ClearML (`scripts/worker_exp61.py`), branch exp/modos-de-falha.

POR QUÊ (exploratório, `casos-carga-e-manutencao-21out`). Os 8 trips aconteceram em carga alta (T5 nas 48 h antes: 7 de 8 acima da mediana, p 0,035; 3 no p90+), o que é físico:
em carga alta, mancal, óleo e selagem ficam mais perto dos setpoints. E, nos 18 episódios do controle, o T5 nas 2 h de notificação ficou no percentil mediano 0,88 nos de trip
contra 0,62 nos FP (AUC 0,72, p 0,067) — a primeira variável a separar FP de acerto depois de seis testes nulos (EXP 36-40). Mas é pós-hoc, 9 contra 9, e da mesma família das
regras de "força" que morreram no teste. Daí este teste nos episódios que eu NÃO olhei.

DESENHO (fixado antes):
  MEDIDA: L = percentil do T5_AVG_A mediano em [nascimento, nascimento + 2 h] contra o T5 da operação estável em [nascimento - 30 d, nascimento) — referência CAUSAL, como seria
  em produção. Secundária: percentil global (fev/2025 a abr/2026, sem rótulo, usa o futuro).
  BASES: atual (já vista: não conta), G168, B800 (ledgers das variantes da sombra) e atual com retreino n = 19.000 — as mesmas do EXP 60. Episódios desde 01/02/2025 (`ci._episodios`);
  TP = intersecta [T - 48 h, T].
  CONJUNTO FORA DA AMOSTRA ("novos"): episódios de G168, B800 e n = 19.000 cujo nascimento não está a menos de 2 h do nascimento de nenhum episódio do controle atual (agrupados,
  um por instante de nascimento distinto).
CRITÉRIOS (fixados antes):
  SINAL REPLICA = AUC(TP > FP) nos "novos" >= 0,65 com p de Mann-Whitney unilateral < 0,10, E AUC >= 0,60 em G168 e em B800 (todos os episódios de cada um).
  NÃO REPLICA no resto.
  SECUNDÁRIO (relatado, não decide): o filtro "suprimir episódio com L < 0,5" em cada base (detecção, FP, horas). Sei de antemão que no atual ele custa o 27/02 (trip de selagem
  em carga 0,44-0,46).
EXPECTATIVAS: AUC da base atual ~0,7 (já visto; não conta); SINAL REPLICA (~30%); o filtro L < 0,5 perde >= 1 detecção em >= 2 das 4 bases (~70%); a medida causal e a global
concordam na direção em todas as bases (~75%).
INTERPRETAÇÃO PREVISTA: REPLICA -> a carga entra como CONTEXTO de prioridade na apresentação ("alarme em carga alta") e como hipótese na sombra; nunca como filtro (custa trip de
selagem em carga baixa). NÃO REPLICA -> o 0,72 era acaso de 9 contra 9; fica só a associação trip x carga como informação para a operação.
LIMITES: os "novos" são poucos e as bases compartilham dado; o T5 é indicador imperfeito de carga (exaustão, também sobe com degradação de combustão); nada se adota sem a sombra.
VERIFICAÇÕES: V1 cada base reproduz as suas referências (atual 7/8 FP 9; G168 8/8 FP 14; B800 8/8 FP 12); V2 a medida da base atual reproduz a exploratória (mediana TP 0,88 e
FP 0,62 na versão global, a +-0,02); V3 todo episódio tem >= 30 amostras estáveis na referência causal.

Uso local (encanamento, sem tabelas):  EXP61_TESTE=1 PYTHONPATH=src .venv/bin/python experiments/61_carga_na_notificacao.py
No worker:  PYTHONPATH=src python scripts/worker_exp61.py --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


E60 = _carrega("exp60", "60_borda_adiada.py")
from nv import io as nvio
from nv.avalia import avalia, carrega_controle
from nv.avalia.metricas import JANELA
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG

TESTE = bool(os.environ.get("EXP61_TESTE"))
ROOT = HERE.parent / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); FIM_REF = pd.Timestamp("2026-04-20", tz="UTC")
NOTIF = pd.Timedelta(hours=2); REF_CAUSAL = pd.Timedelta(days=30); PERTO = pd.Timedelta(hours=2)


def episodios(ci, al, trips):
    out = []
    for a, b in ci._episodios(al.loc[INICIO8:]):
        out.append(dict(inicio=a, fim=b, tipo="TP" if any(a <= t and b >= t - JANELA for t in trips) else "FP"))
    return pd.DataFrame(out)


def medida(t5, a):
    w = t5.loc[a:a + NOTIF].dropna(); r = t5.loc[a - REF_CAUSAL:a - pd.Timedelta(minutes=2)].dropna()
    if len(w) == 0 or len(r) < 30:
        return np.nan, np.nan, len(r)
    x = float(w.median()); return float((r < x).mean()), float((GL["ref_global"] < x).mean()), len(r)


GL = {}


def auc(df, col):
    a = df[df.tipo == "TP"][col].dropna(); b = df[df.tipo == "FP"][col].dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan, len(a), len(b)
    u = stats.mannwhitneyu(a, b, alternative="greater"); return float(u.statistic / (len(a) * len(b))), float(u.pvalue), len(a), len(b)


def main():
    t0 = time.time(); ci = _ci(); E60.GL["ci"] = ci; ok = {}
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]
    trips = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; E60.GL["trips"] = trips
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"]; t5 = g["T5_AVG_A"].where(est)
    GL["ref_global"] = t5.loc[INICIO8:FIM_REF].dropna().to_numpy()
    B = {}
    for v in ("atual", "G168", "B800"):
        B[v] = carrega_controle(v)
    E60.GL["idx"] = B["atual"][0].index
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    B["atual n=19000"] = (E60.base_n(ci, det, proc, 19000).reindex(E60.GL["idx"]).fillna(False), B["atual"][1])
    for v, (dr, fr) in E60.REF.items():
        m = E60.mede(*B[v]); ok[f"V1_{v}"] = (m["det"], m["fp"]) == (dr, fr)
    print(f"V1 — bases reproduzem as referências: {all(ok.values())}", flush=True)
    E = {}
    for v, (al, vig) in B.items():
        d = episodios(ci, al, trips); vals = [medida(t5, a) for a in d.inicio]
        d["L_causal"] = [x[0] for x in vals]; d["L_global"] = [x[1] for x in vals]; d["n_ref"] = [x[2] for x in vals]; d["base"] = v; E[v] = d
    at = E["atual"]; mtp = float(at[at.tipo == "TP"].L_global.median()); mfp = float(at[at.tipo == "FP"].L_global.median())
    ok["V2"] = abs(mtp - 0.88) <= 0.02 and abs(mfp - 0.62) <= 0.02
    print(f"V2 — base atual, medida global: mediana TP {mtp:.2f} (0,88), FP {mfp:.2f} (0,62) -> {'PASSA' if ok['V2'] else 'FALHA'}", flush=True)
    todos = pd.concat(E.values()); ok["V3"] = bool((todos.n_ref >= 30).all())
    print(f"V3 — referência causal com >= 30 amostras em todos os {len(todos)} episódios: {'PASSA' if ok['V3'] else 'FALHA'} (mínimo {int(todos.n_ref.min())})", flush=True)
    nasc_atual = at.inicio.tolist()
    novos = todos[(todos.base != "atual") & todos.inicio.map(lambda a: all(abs(a - x) > PERTO for x in nasc_atual))].copy()
    novos = novos.sort_values("inicio").drop_duplicates(subset=["inicio"])
    GL.update(E=E, novos=novos, B=B, ci=ci)
    if TESTE:
        with contextlib.redirect_stdout(io.StringIO()):
            relata(ok, t0, escreve=False)
        print(f"TESTE: encanamento e relatório ok em {time.time() - t0:.0f} s (saída suprimida, nada gravado). {ok}", flush=True); return
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    relata(ok, t0, escreve=True)


def relata(ok, t0, escreve):
    E, novos, B, ci = GL["E"], GL["novos"], GL["B"], GL["ci"]
    loc = lambda t: (t - pd.Timedelta(hours=3)).strftime("%d/%m/%Y %H:%M")
    print("\nSEPARAÇÃO POR BASE (AUC TP > FP, p unilateral; n TP / n FP)", flush=True)
    for v, d in E.items():
        a1 = auc(d, "L_causal"); a2 = auc(d, "L_global")
        print(f"   {v:14s} causal: AUC {a1[0]:.2f} (p {a1[1]:.3f}; {a1[2]}/{a1[3]}) | global: AUC {a2[0]:.2f} (p {a2[1]:.3f})" + ("   <- já vista, não conta" if v == "atual" else ""), flush=True)
    an = auc(novos, "L_causal"); ang = auc(novos, "L_global")
    print(f"   NOVOS (fora da amostra) causal: AUC {an[0]:.2f} (p {an[1]:.3f}; {an[2]} TP / {an[3]} FP) | global: AUC {ang[0]:.2f} (p {ang[1]:.3f})", flush=True)
    print("\nEPISÓDIOS NOVOS (nascimento local, tipo, base, L causal, L global)", flush=True)
    for r in novos.itertuples():
        print(f"   {loc(r.inicio)} {r.tipo} {r.base:14s} L causal {r.L_causal:.2f} | global {r.L_global:.2f}", flush=True)
    print("\nSECUNDÁRIO — filtro 'suprimir episódio com L causal < 0,5' (det/8 | FP | horas)", flush=True)
    for v, (al, vig) in B.items():
        d = E[v]; a = al.copy()
        for r in d[d.L_causal < 0.5].itertuples():
            a.loc[r.inicio:r.fim] = False
        m0 = E60.mede(al, vig); m1 = E60.mede(a, vig)
        print(f"   {v:14s} {m0['det']}/8 FP {m0['fp']:2d} {m0['horas']:5.0f} h -> {m1['det']}/8 FP {m1['fp']:2d} {m1['horas']:5.0f} h | perdidos: {[e for e in m0['ev'] if m0['ev'][e] and not m1['ev'][e]]}", flush=True)
    g1 = auc(E["G168"], "L_causal")[0]; g2 = auc(E["B800"], "L_causal")[0]
    rep = (an[0] >= 0.65 and an[1] < 0.10 and g1 >= 0.60 and g2 >= 0.60)
    print(f"\nCLASSIFICAÇÃO: novos AUC {an[0]:.2f} (p {an[1]:.3f}), G168 {g1:.2f}, B800 {g2:.2f} -> {'SINAL REPLICA' if rep else 'NÃO REPLICA'}", flush=True)
    if escreve:
        pd.concat(E.values()).to_csv(ROOT / "exp61_episodios.csv", index=False)
    print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

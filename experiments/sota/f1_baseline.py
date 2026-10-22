#!/usr/bin/env python3
"""SOTA FASE 1 — BASELINE REPRODUTÍVEL DO PCA v2 no protocolo único (`experiments/sota/protocolo.py`). Tag ClearML `sota-baseline`.

Fixado em 22/10/2026, antes de rodar. DESCRITIVO: não há hipótese; é a linha de comparação de todas as fases seguintes.
BASES: atual (o ledger do controle, replay causal semanal), G168 e B800 (variantes da sombra) e atual com retreino n = 19.000 (o ruído de retreino do t e do p).
MEDIDAS: todas as do protocolo, com IC 95% (eventos por bootstrap de eventos, custo por blocos mensais); a comparação pareada de cada variante contra a atual;
ESTABILIDADE DO SCORE EM OPERAÇÃO NORMAL (o brief pede, porque há poucos eventos): por mês e por canal, a fração do tempo vigiado com o canal acima do limiar A, longe dos
trips (fora de [T − 7 d, T + 2 d]), com o coeficiente de variação entre meses; TEMPO DE INFERÊNCIA: `preprocessar` + `prever` numa janela de 75 d (a execução semanal de
produção), em segundos por dia de dado.
VERIFICAÇÃO: V1 cada base reproduz a referência (atual 7/8 FP 9; G168 8/8 FP 14; B800 8/8 FP 12; n=19.000 5/8 FP 11).

Uso local:  PYTHONPATH=src:experiments .venv/bin/python -m sota.f1_baseline
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp f1_baseline --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv.avalia import carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from sota import protocolo as P

ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "f1_baseline"
REF = {"atual": (7, 9), "G168": (8, 14), "B800": (8, 12), "n=19000": (5, 11)}


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, EXP / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


def bases(ci, det):
    E60 = _carrega("exp60", "60_borda_adiada.py"); E60.GL["ci"] = ci
    B = {v: carrega_controle(v) for v in ("atual", "G168", "B800")}
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    B["n=19000"] = (E60.base_n(ci, det, proc, 19000).reindex(B["atual"][0].index).fillna(False), B["atual"][1])
    return B, proc


def estabilidade(proc, det, trips):
    """Fração mensal do tempo vigiado com cada canal acima do limiar A, longe dos trips."""
    longe = pd.Series(True, index=proc.index)
    for t in trips:
        longe.loc[t - pd.Timedelta(days=7):t + pd.Timedelta(days=2)] = False
    m = proc["mask"] & longe & (proc.index >= P.INICIO) & (proc.index <= P.FIM); linhas = []
    for c in ("t", "p", "sp", "vb"):
        thr = det["base"][c] * det["k_nivel_a"][c]; acima = (proc[c] > thr) & m
        g = pd.DataFrame({"acima": acima[m], "mes": proc.index[m].tz_localize(None).to_period("M")}).groupby("mes")["acima"].mean()
        linhas += [dict(canal=c, mes=str(k), frac_acima=float(v)) for k, v in g.items()]
    return pd.DataFrame(linhas)


def tempo_inferencia(ci, modelos):
    csv = next((PKG / "dados" / "2025_2026").glob("data_*_raw.csv"))
    with contextlib.redirect_stdout(io.StringIO()):
        df = ci.preparar_grade(ci.carregar_dados(csv))
    w = df.loc[df.index[-1] - pd.Timedelta(days=75):]; t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        out = ci.preprocessar(modelos, w); ci.prever(modelos, out)
    return (time.perf_counter() - t0) / 75.0


def main():
    t0 = time.time(); ci = _ci(); SAIDA.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; trips = P.trips_alvo(carrega_registro())
    B, proc = bases(ci, det); vig = B["atual"][1]; M, I = {}, {}
    for v, (al, vg) in B.items():
        M[v] = P.mede(al, vg, trips); I[v] = P.ic(al, vg, trips)
        ok = (M[v]["det"], M[v]["fp"]) == REF[v]
        print(f"V1 [{v}] {M[v]['det']}/8 FP {M[v]['fp']} (referência {REF[v][0]}/8 FP {REF[v][1]}) -> {'PASSA' if ok else 'FALHA'}", flush=True)
        if not ok:
            print("verificação falhou — parando.", flush=True); return
    print(f"\nBASELINE PCA v2 — protocolo único ({P.INICIO:%d/%m/%Y} a {P.FIM:%d/%m/%Y}, {len(trips)} trips, janela 48 h; IC 95%)", flush=True)
    f = lambda x, c: f"[{c[0]:.0f}-{c[1]:.0f}]" if x == "int" else f"[{c[0]:.2f}-{c[1]:.2f}]"
    for v in B:
        m, c = M[v], I[v]
        print(f" [{v}] detecção {m['det']}/8 {f('int', c['det'])} | início {m['inicio']} {f('int', c['inicio'])} | banda {m['banda']} {f('int', c['banda'])} | "
              f"FP {m['fp']} | FP/mês {m['fp_mes']:.3f} {f('', c['fp_mes'])} | FP/dia {m['fp_dia']:.4f} | horas {m['horas']:.0f} ({m['horas_mes']:.1f}/mês {f('', c['horas_alarme_730h'])}) | fora {m['fora']:.0f}", flush=True)
        print(f"        lead médio {m['lead_medio']:.1f} h | mediano {m['lead_mediano']:.1f} h {f('', c['lead_mediano'])} | {m['n_censurado']} de {m['det']} censurados em 48 h | "
              f"F1 por evento {m['f1']:.2f} (precisão {m['precisao']:.2f}, recall {m['recall']:.2f}, {m['episodios']} episódios)", flush=True)
    print("\nCOMPARAÇÃO PAREADA contra a base atual (regra: detecção só com |Δ| >= 2 e IC sem zero; custo só com IC sem zero)", flush=True)
    for v in ("G168", "B800", "n=19000"):
        d = P.compara(B[v][0], B["atual"][0], vig, trips)
        print(f"   {v:8s} Δdet {d['d_det']:+d} [{d['ic_d_det'][0]:+.0f}, {d['ic_d_det'][1]:+.0f}] | ΔFP/mês {d['d_fp_mes']:+.3f} [{d['ic_d_fp_mes'][0]:+.2f}, {d['ic_d_fp_mes'][1]:+.2f}] | "
              f"Δhoras/mês {d['d_horas_mes']:+.1f} [{d['ic_d_horas_mes'][0]:+.1f}, {d['ic_d_horas_mes'][1]:+.1f}] -> "
              f"{'difere' if (d['difere_det'] or d['difere_fp'] or d['difere_horas']) else 'indistinguível'} (det {d['difere_det']}, FP {d['difere_fp']}, horas {d['difere_horas']})", flush=True)
    E = estabilidade(proc, det, trips); E.to_csv(SAIDA / "estabilidade_mensal.csv", index=False)
    print("\nESTABILIDADE EM OPERAÇÃO NORMAL (base atual; fração do tempo vigiado acima do limiar A, longe dos trips)", flush=True)
    for c, g in E.groupby("canal", sort=False):
        x = g.frac_acima.to_numpy(); print(f"   {c:3s} mediana mensal {np.median(x):.3f} | mín {x.min():.3f} | máx {x.max():.3f} | CV entre meses {x.std() / x.mean():.2f}", flush=True)
    ti = tempo_inferencia(ci, modelos); print(f"\nTEMPO DE INFERÊNCIA (preprocessar + prever, janela de 75 d): {ti:.3f} s por dia de dado", flush=True)
    pd.DataFrame([dict(base=v, **M[v], **{f"ic_{k}": c for k, c in I[v].items()}) for v in B]).to_csv(SAIDA / "baseline_pca_v2.csv", index=False)
    print(f"\ntempo total {time.time() - t0:.0f} s; tabelas em {SAIDA}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SOTA H3 — TranAD (multivariado, dependência entre sensores) NO LUGAR DO PCA nos canais t e p do v2, um tiro pré-registrado. Tag ClearML `sota-h3`.

PRÉ-REGISTRO (22/10/2026), COMMITADO ANTES DE RODAR. Hipótese do brief: um modelo multivariado com atenção capta anomalias contextuais e dá menos FP. O PCA do v2 já é
multivariado; a pergunta aqui é se o TranAD (transformer com treino adversarial em duas fases) supera o PCA no MESMO lugar: o escore do canal t (14 temperaturas) e do
canal p (12 pressões), com a mesma decisão do v2 (sp, vb, limiares, voto, CUSUM, refratário inalterados). Já testados e sem ganho sobre o PCA: K-Means, análise fatorial,
LSTM-AE (EXP 44), 300 configurações com LOEO (EXP 46), CNN-1D sequencial (EXP 11), modelo por grupo (EXP 50). O TranAD nunca foi testado.
IMPLEMENTAÇÃO JÁ TESTADA: as classes do modelo do TSB-AD 1.5 (que adapta o repositório oficial), copiadas sem alteração em `sota/tranad_modelo.py`; o wrapper escala pelo
min-max do treino e atribui o escore ao fim de cada janela (causal). Janela de 10 (o padrão do repositório oficial: 20 min na grade de 2 min).
PROTOCOLO DE TREINO (o do harness fiel, `nv.walk.walk_forward`): retreino no dia 1 de cada mês de fev/2025 a abr/2026 com as últimas 20.000 amostras ESTÁVEIS anteriores
(janelas de 10 instantes consecutivos todos estáveis), escore do mês dividido pelo p99 do escore do próprio treino, EWMA do canal, decisão do `prever`. 3 sementes.
VARIANTES: PRINCIPAL = TranAD em t E p; secundárias = só t, só p.
MEDIDAS: o protocolo único (detecção, início, banda, FP, FP/mês, horas, lead censurado, F1, IC) e a comparação pareada contra o PCA v2 (o ledger do controle, base atual).
CLASSIFICAÇÃO (por semente, na variante principal; fixada antes): MELHOR = FP ou horas com queda DECLARADA (IC pareado sem o zero) e detecção sem queda declarada e Δdet >= −1;
PIOR = queda declarada de detecção, ou FP ou horas com alta declarada; EQUIVALENTE no resto. VEREDITO = a classe de >= 2 das 3 sementes; sem maioria, INSTÁVEL.
EXPECTATIVAS: V1-V2 passam (~90%); EQUIVALENTE ~45%, PIOR ~45%, MELHOR ~10%; o piso de ruído entre sementes cobre >= 1 detecção (~70%).
LIMITES: 8 eventos; hiperparâmetros do TSB-AD/oficial, sem busca; CPU; 3 sementes.
VERIFICAÇÕES: V1 o MESMO laço de walk-forward deste arquivo, com o PCA fiel e janela 1, reproduz os escores do harness (`escores_sub34`, t e p com n = 20.000) com
diferença <= 1e-9; V2 nenhuma janela de treino tem instante no mês pontuado ou depois.

RESULTADO (22/10/2026; worker do ClearML, tarefa df6909b4cf96429c8c4b3513549c424a; pré-registro 2fc1d4c). V1-V2 PASSAM. EQUIVALENTE pela regra — e pior em ponto em todas
as sementes; nenhum ganho.
    PCA v2 (atual)          7/8 | início 6 | FP  9 (0,807/mês) | 1.457 h | F1 0,64
    TranAD t+p, sementes    5/8, 6/8, 4/8 | FP 13, 13, 13 (1,166/mês) | 1.476, 1.366, 1.682 h | F1 0,38, 0,44, 0,32 -> EQUIVALENTE, EQUIVALENTE, PIOR (Δdet −3 [−6; −1])
    TranAD só t             6/8, 6/8, 5/8 | FP 13 | ΔFP/mês +0,36 [+0,09; +0,66] (alta DECLARADA)
    TranAD só p             5/8, 7/8, 6/8 | FP 13, 12, 12
  Em todas as sementes e variantes: detecção igual ou menor (−0 a −3), FP maior (+3 a +4); horas parecidas. O ruído entre sementes cobre até 2 detecções (4/8 a 6/8 na t+p).
PLACAR: ACERTEI: V1-V2 (~90%); EQUIVALENTE (~45%); o ruído entre sementes cobre >= 1 detecção (~70%). (Dei ~45% a PIOR e ~10% a MELHOR.)
VEREDITO (como pré-registrado): EQUIVALENTE -> o TranAD não substitui o PCA; a H3 do brief não se confirma aqui (como K-Means, FA, LSTM-AE, CNN-1D e Optuna antes dele).

Uso local (encanamento: 2 meses, 1 semente, 2 épocas):  SOTA_TESTE=1 PYTHONPATH=src:experiments .venv/bin/python -m sota.h3_tranad
No worker:  PYTHONPATH=src python scripts/worker_sota.py --exp h3_tranad --tags sota-h3 --procs 24 --enqueue --queue default
"""
from __future__ import annotations
import contextlib, importlib.util, io, os, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; RAIZ = EXP.parent
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(EXP))
from nv import io as nvio
from nv.avalia import carrega_controle
from nv.causal import carrega_registro
from nv.controle_sinais import _ci, PKG
from nv.plugin import decide_v2, substitui
from sota import protocolo as P
from sota.tranad_modelo import TranADCausal

TESTE = bool(os.environ.get("SOTA_TESTE"))
ROOT = RAIZ / "data_local"; SAIDA = ROOT / "sota" / "h3_tranad"; N_TREINO = 20000; W = 10
MESES = pd.date_range("2025-02-01", "2026-04-01", freq="MS", tz="UTC")[: (2 if TESTE else None)]
SEMENTES = (0,) if TESTE else (0, 1, 2)
GL = {}


def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, EXP / arq); m = importlib.util.module_from_spec(spec); sys.modules[nome] = m; spec.loader.exec_module(m); return m


def janelas(X, fins, w):
    return np.stack([X[e - w + 1:e + 1] for e in fins]) if len(fins) else np.empty((0, w, X.shape[1]))


def caminha(arg):
    """Walk-forward mensal de um canal: (canal, modelo, semente). modelo 'pca' (janela 1, V1) ou 'tranad'."""
    canal, qual, semente = arg; g, est, tags = GL["g"], GL["est"], GL["tags"][canal]; X = g[tags].to_numpy("float64"); idx = g.index
    fin = np.isfinite(X).all(axis=1); w = 1 if qual == "pca" else W; out = np.full(len(X), np.nan); v2 = True
    for k, m0 in enumerate(MESES):
        m1 = MESES[k + 1] if k + 1 < len(MESES) else P.FIM + pd.Timedelta(minutes=2)
        i0, i1 = idx.searchsorted(m0), idx.searchsorted(m1)
        cand = np.flatnonzero(est[:i0] & fin[:i0])
        if len(cand) < N_TREINO // 4:
            continue
        cand = cand[-N_TREINO:]
        if w == 1:
            fins_tr = cand
        else:
            ok = np.zeros(len(X), bool); ok[cand] = True
            fins_tr = np.array([e for e in cand if e - w + 1 >= 0 and ok[e - w + 1:e + 1].all()])
        v2 &= bool(len(fins_tr) == 0 or fins_tr.max() < i0)
        Wt = janelas(X, fins_tr, w)
        if qual == "pca":
            mod = GL["fab_pca"](semente=semente).fit(Wt[:, 0, :]); sc = lambda Z: mod.score(Z[:, 0, :])
        else:
            mod = TranADCausal(len(tags), W, semente=semente + 1000 * k, epocas=2 if TESTE else 50).fit(Wt); sc = mod.score
        p99 = float(np.nanquantile(sc(Wt), 0.99))
        fins_te = np.array([e for e in range(i0, i1) if e - w + 1 >= 0 and fin[e - w + 1:e + 1].all()])
        if len(fins_te):
            out[fins_te] = sc(janelas(X, fins_te, w)) / max(p99, 1e-12)
    return canal, qual, semente, pd.Series(out, index=idx), v2


def main():
    t0 = time.time(); SAIDA.mkdir(parents=True, exist_ok=True); ok = {}; ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos")
    det = modelos[-1]["detector"]; E46 = _carrega("exp46", "46_optuna_modelos_loeo_placebo.py")
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool)
    tags = {"t": list(modelos[-1]["normalizacao"]["temperatura"]["cols"]), "p": list(modelos[-1]["normalizacao"]["pressao"]["cols"])}
    GL.update(g=g, est=est, tags=tags, fab_pca=lambda semente=0: E46.fabrica({"fam": "pca", "var": 0.95})(semente=semente))
    jobs = [(c, "pca", 0) for c in ("t", "p")] + [(c, "tranad", s) for s in SEMENTES for c in ("t", "p")]
    with Pool(min(int(os.environ.get("SOTA_PROCS", 8)), len(jobs))) as pool:
        R = {(c, q, s): (ser, v2) for c, q, s, ser, v2 in pool.imap_unordered(caminha, jobs, chunksize=1)}
    for c in ("t", "p"):
        ref = pd.read_parquet(ROOT / "escores_sub34" / f"{c}_A_{N_TREINO}.parquet")["s"]; s = R[(c, "pca", 0)][0]
        j = ref.index.intersection(s.index); j = j[(j >= MESES[0]) & (j < (MESES[-1] + pd.offsets.MonthBegin(1)))]
        a, b = ref.loc[j].to_numpy(), s.loc[j].to_numpy(); m = np.isfinite(a) & np.isfinite(b)
        ok[f"V1_{c}"] = bool(m.any() and np.max(np.abs(a[m] - b[m])) <= 1e-9 and (np.isfinite(a) == np.isfinite(b)).all())
        print(f"V1 [{c}] laço deste arquivo com PCA fiel x escores_sub34: dif. máx {np.max(np.abs(a[m] - b[m])):.1e}, NaN divergentes {int((np.isfinite(a) != np.isfinite(b)).sum())} -> {'PASSA' if ok[f'V1_{c}'] else 'FALHA'}", flush=True)
    ok["V2"] = all(v for _, v in R.values()); print(f"V2 — nenhuma janela de treino no mês pontuado ou depois -> {'PASSA' if ok['V2'] else 'FALHA'} ({time.time() - t0:.0f} s)", flush=True)
    if not all(ok.values()):
        print(f"verificação falhou — parando. {ok}", flush=True); return
    proc = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet")[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool)
    al_pca, vig = carrega_controle("atual"); trips = P.trips_alvo(carrega_registro()); linhas = []
    ew = lambda s, c: s.ewm(halflife=pd.Timedelta(det["halflife"][c]), times=s.index).mean().where(s.notna())
    for s in SEMENTES:
        for var, canais in (("t+p", ("t", "p")), ("só t", ("t",)), ("só p", ("p",))):
            px = proc.copy()
            for c in canais:
                px = substitui(px, c, ew(R[(c, "tranad", s)][0], c))
            with contextlib.redirect_stdout(io.StringIO()):
                al = decide_v2(px, det)["is_anomaly"].astype(bool)
            al = al.reindex(vig.index).fillna(False); al.loc[:MESES[0] - pd.Timedelta(minutes=2)] = al_pca.loc[:MESES[0] - pd.Timedelta(minutes=2)]
            n = 400 if TESTE else P.N_BOOT; m = P.mede(al, vig, trips); c_ = P.ic(al, vig, trips, n=n); d = P.compara(al, al_pca, vig, trips, n=n)
            linhas.append(dict(semente=s, variante=var, **m, ic_det=c_["det"], ic_fp_mes=c_["fp_mes"], **{k: v for k, v in d.items()}))
    T = pd.DataFrame(linhas); GL.update(T=T, mp=P.mede(al_pca, vig, trips))
    if TESTE:
        print(f"TESTE: encanamento ok em {time.time() - t0:.0f} s ({len(T)} linhas; saída suprimida).", flush=True); return
    relata(); T.to_csv(SAIDA / "h3_tranad.csv", index=False); print(f"\nverificações: {ok} | tempo total {time.time() - t0:.0f} s", flush=True)


def relata():
    T, mp = GL["T"], GL["mp"]
    print(f"\nPCA v2 (ledger, base atual): {mp['det']}/8 | início {mp['inicio']} | banda {mp['banda']} | FP {mp['fp']} ({mp['fp_mes']:.3f}/mês) | {mp['horas']:.0f} h | F1 {mp['f1']:.2f}", flush=True)
    print("TranAD no lugar do PCA (comparação pareada contra o PCA v2)", flush=True)
    for r in T.itertuples():
        print(f"   semente {r.semente} {r.variante:5s} {r.det}/8 [{r.ic_det[0]:.0f}-{r.ic_det[1]:.0f}] início {r.inicio} banda {r.banda} | FP {r.fp:2d} ({r.fp_mes:.3f}/mês) | {r.horas:5.0f} h | "
              f"lead med. {r.lead_mediano:.1f} h ({r.n_censurado} cens.) | F1 {r.f1:.2f} | Δdet {r.d_det:+d} [{r.ic_d_det[0]:+.0f}, {r.ic_d_det[1]:+.0f}] ΔFP/mês {r.d_fp_mes:+.2f} "
              f"[{r.ic_d_fp_mes[0]:+.2f}, {r.ic_d_fp_mes[1]:+.2f}] Δh/mês {r.d_horas_mes:+.1f} [{r.ic_d_horas_mes[0]:+.1f}, {r.ic_d_horas_mes[1]:+.1f}]", flush=True)
    cls = []
    for r in T[T.variante == "t+p"].itertuples():
        pior = (r.difere_det and r.d_det < 0) or (r.difere_fp and r.d_fp_mes > 0) or (r.difere_horas and r.d_horas_mes > 0)
        melhor = ((r.difere_fp and r.d_fp_mes < 0) or (r.difere_horas and r.d_horas_mes < 0)) and not (r.difere_det and r.d_det < 0) and r.d_det >= -1
        cls.append("PIOR" if pior else "MELHOR" if melhor else "EQUIVALENTE")
    v = max(set(cls), key=cls.count); vered = v if cls.count(v) >= 2 else "INSTÁVEL"
    print(f"\nCLASSIFICAÇÃO por semente (t+p): {cls} -> VEREDITO {vered}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXP 15 — TRILHA B: PREVISOR DE SÉRIE COMO MODELO DE RESÍDUO (o caminho 1 do AutoKeras, à mão e sem busca de arquitetura).

PRÉ-REGISTRO (15/10/2026), COMMITADO ANTES DE RODAR.

PERGUNTA. Os modelos de resíduo do EXP 14 previam cada tag a partir das OUTRAS tags no MESMO instante (ridge, GBM, PCA) e os flexíveis perderam detecção (3-4/7 contra 7/7 do PCA de k = 1). Um resíduo PREDITIVO TEMPORAL (prever o vetor de tags em i a partir da janela anterior) traz informação diferente, a dinâmica? Mesma agregação do v2 (máximo por sensor normalizado pelo p99), por família (temperatura 14 tags; pressão 12), retreino mensal causal nas últimas 20.000 amostras-alvo estáveis cuja janela inteira é estável e finita, EWMA 1 h, mapa de quantis e `ci.prever` como caixa-preta.
GRADE FIXA (sem busca; não é NAS): janela L = 15 amostras (30 min); antecedência H em {1, 15, 90} passos (2 min, 30 min, 3 h); modelos persist (controle trivial: o resíduo é a variação em H passos), ar (ridge sobre a janela achatada) e gru (GRU(32), 12 épocas, previsão = última amostra + correção aprendida; H em {15, 90}, sementes 0, 1, 2). Piso de ruído: treino de 19.000 e 21.000 em persist/ar; 3 sementes nos gru.
DOIS USOS, cada um contra a sua referência NO MESMO HARNESS (o EXP 14 mostrou que o harness por família não reproduz o bundle: o pca1 em t dá 7/7 com FP 15 contra 9, e por isso a referência certa é o pca1 recalculado aqui, não o controle):
  (S) SUBSTITUIÇÃO do canal t (famílias de temperatura) ou p (pressão); referência = pca1 do EXP 14 (t: 7/7, FP 15, 590 h fora; p: 6/7, FP 12, 754 h fora).
  (Q) QUINTO canal (só temperatura, voto V1 (3, 2), base e k do t, como no EXP 09/11); referência = AE pontual D0_20k em 10 sementes (7/7, FP 12, 563 h fora).
CRITÉRIOS (fixados antes). SOBRE A REFERÊNCIA (S): GANHO = detecção >= a da referência E FP <= referência - 3 E horas fora <= 0,8 x a referência (acima do piso de ruído da régua nos dois custos). Contra o controle: PROMISSOR = detecção >= 5/7, FP <= 12 e horas fora <= 800 h; DIGNO = >= 6/7, FP <= 9, <= 407 h. (Q): MELHOR = detecção >= 7/7 com FP <= 10 e horas fora <= 450 (o critério do EXP 11); IGUAL = >= 6/7 com FP 11 a 14 e <= 700 h; PIOR = o resto.
EXPECTATIVAS: persist, em qualquer H, fica <= 3/7 em S de t (~85%: o resíduo da variação de 2 min a 3 h é ruído e transiente, não deriva); ar com H = 1 <= 4/7 (~75%); algum previsor com H = 90 ou 15 chega a >= 5/7 em S de t (~35%); gru melhor que ar em detecção em pelo menos metade das células (~40%); algum GANHO sobre a referência em S (~7%); algum MELHOR em Q (~8%); algum PROMISSOR em S no canal p (~35%, a pressão é mais fácil: o EXP 14 deu 6/7 a vários).
INTERPRETAÇÃO PREVISTA. Se nada passa, a dinâmica local (2 min a 3 h) não carrega o precursor além do que o PCA rígido já pega, e o AutoKeras `TimeseriesForecaster` (que buscaria sobre esta mesma classe) não tem o que procurar: fecha-se o caminho 1. Se algo passa, entra o episódio por episódio (13b) e o placebo antes de qualquer ideia de busca de capacidade.
LIMITES. 7 eventos; uma classe de previsor e uma grade pequena; GRU com 12 épocas e sem warm-start (o piso de ruído neural é maior); o mapa de quantis é global; herda limiares sem recalibrar.

Uso:  PYTHONPATH=src .venv/bin/python experiments/15_previsor_residual.py
"""
from __future__ import annotations
import contextlib, copy, importlib.util, io, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
def _carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, HERE / arq); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
E4 = _carrega("exp04", "04_escada_de_baselines.py"); E5 = _carrega("exp05", "05_decomposicao_do_v2.py"); E6 = _carrega("exp06", "06_plugin_de_canais.py")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle
from nv.controle_sinais import sinais_causais, _ci, PKG
from nv.modelos.previsor import walk_previsor
from nv.plugin import decide_v2, mapeia_quantis, substitui
from nv.politica import ewma_vigiado

INICIO = E4.INICIO; ROOT = Path(__file__).resolve().parents[1] / "data_local"; CACHE = ROOT / "escores_prev"; GL = {}
FAMILIAS = {"t": "temperatura", "p": "pressao"}


def tarefa_escore(a):
    kind, H, canal, n_treino, sem = a; f = CACHE / f"{canal}_{kind}_H{H}_{n_treino}_{sem}.parquet"
    if f.exists():
        return a, pd.read_parquet(f)["s"]
    import torch
    torch.set_num_threads(2); t0 = time.time()
    s = walk_previsor(GL["g"], GL["tags"][canal], GL["est"], pd.Timestamp("2025-03-01", tz="UTC"), GL["fim"], kind, H, n_max=n_treino, semente=sem)
    CACHE.mkdir(parents=True, exist_ok=True); s.to_frame("s").to_parquet(f); print(f"      {canal} {kind} H={H} n={n_treino} s={sem}: {time.time() - t0:.0f} s", flush=True)
    return a, s


def resume(ms):
    d = [m["det"] for m in ms]; f = [m["fp"] for m in ms]; o = [m["fora"] for m in ms]
    return np.median(d), min(d), max(d), np.median(f), min(f), max(f), np.median(o), np.median([m["horas"] for m in ms]), np.nanmedian([m["lead"] for m in ms])


def avalia_s(e, canal, tag_proc, det, reg, vig_g):
    novo = mapeia_quantis(e, GL["proc"][canal], GL["vigS"]); r = E6.mede(decide_v2(substitui(GL["proc"], canal, novo), det), reg)
    return dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h)


def avalia_q(e, det, reg):
    d2 = copy.deepcopy(det); d2["sinais"] = list(det["sinais"]) + ["ae"]
    for k in ("base", "k_nivel_a", "k_nivel_b"):
        d2[k]["ae"] = det[k]["t"]
    d2["voto_nivel_a"], d2["voto_nivel_b"] = 3, 2
    p2 = GL["proc"].copy(); p2["ae"] = mapeia_quantis(e, GL["proc"]["t"], GL["vigS"]); r = E6.mede(decide_v2(p2, d2), reg)
    return dict(det=r.det, fp=r.fp, fora=r.horas_alarme_fora, horas=r.horas_alarme, lead=r.lead_medio_h)


def main():
    g = nvio.carrega_grade(); est = nvio.mascaras(g)["estavel"].to_numpy(bool); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tags = {c: list(modelos[-1]["normalizacao"][f]["cols"]) for c, f in FAMILIAS.items()}
    al_c, vig_c = carrega_controle("atual"); fim = vig_c.index[-1]; reg = [t for t in carrega_registro() if t >= INICIO]; ctrl = avalia(al_c, vig_c, reg, inicio_periodo=INICIO)
    S = sinais_causais(E5.CORTE0, cache=E5.CACHE); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); vig_g = vig_c.reindex(g.index).fillna(False)
    GL.update(g=g, est=est, fim=fim, tags=tags, proc=proc, vigS=proc["mask"])
    pedidos = []
    for canal in FAMILIAS:
        for H in (1, 15, 90):
            pedidos += [("persist", H, canal, 20000, 0)] + [("ar", H, canal, nt, 0) for nt in (20000, 19000, 21000)]
        for H in (15, 90):
            pedidos += [("gru", H, canal, 20000, s) for s in (0, 1, 2)]
    print(f"CONTROLE: {E6.linha(ctrl)}\n{len(pedidos)} escores (previsores por família)", flush=True)
    with Pool(4) as p:
        R = dict(p.map(tarefa_escore, pedidos, chunksize=1))
    EW = {a: ewma_vigiado(s, vig_g, "1h").reindex(proc.index) for a, s in R.items()}
    # referências no mesmo harness
    print("\nREFERÊNCIAS (mesmo harness): pca1 do EXP 14 e AE pontual D0_20k", flush=True)
    refs = {}
    for canal in FAMILIAS:
        ms = [avalia_s(ewma_vigiado(pd.read_parquet(ROOT / "escores_fam" / f"{canal}_pca1_{nt}_0.parquet")["s"], vig_g, "1h").reindex(proc.index), canal, None, det, reg, vig_g) for nt in (20000, 19000, 21000)]
        refs[canal] = resume(ms); r = refs[canal]; print(f"   S {canal} <- pca1: detecção {r[0]:.0f}/7 ({r[1]:.0f}-{r[2]:.0f}) | FP {r[3]:.0f} | fora {r[6]:.0f} h | totais {r[7]:.0f} h")
    msq = [avalia_q(ewma_vigiado(pd.read_parquet(ROOT / "escores_ae" / f"D0_20k_temperatura_{s}.parquet")["s"], vig_g, "1h").reindex(proc.index), det, reg) for s in range(10)]
    rq = resume(msq); print(f"   Q AE pontual (10 sementes): detecção {rq[0]:.0f}/7 ({rq[1]:.0f}-{rq[2]:.0f}) | FP {rq[3]:.0f} | fora {rq[6]:.0f} h | totais {rq[7]:.0f} h", flush=True)
    print("\nPREVISORES (mediana das variantes | faixa de detecção | FP | horas fora | totais | lead)", flush=True)
    grupos = {}
    for (kind, H, canal, nt, sm) in R:
        grupos.setdefault((kind, H, canal), []).append((kind, H, canal, nt, sm))
    for (kind, H, canal), chaves in grupos.items():
        ref = refs[canal]
        ms = [avalia_s(EW[k], canal, None, det, reg, vig_g) for k in chaves]; r = resume(ms)
        ganho = r[0] >= ref[0] and r[3] <= ref[3] - 3 and r[6] <= 0.8 * ref[6]
        ver = "DIGNO" if r[0] >= 6 and r[3] <= 9 and r[6] <= 407 else "PROMISSOR" if r[0] >= 5 and r[3] <= 12 and r[6] <= 800 else "-"
        print(f"   S {canal} <- {kind:7s} H={H:<3d} ({len(ms)} variantes): detecção {r[0]:.0f}/7 ({r[1]:.0f}-{r[2]:.0f}) | FP {r[3]:.0f} ({r[4]:.0f}-{r[5]:.0f}) | fora {r[6]:.0f} h | totais {r[7]:.0f} h | lead {r[8]:.1f} h -> {ver}{' + GANHO sobre pca1' if ganho else ''}", flush=True)
        if canal == "t":
            mq = [avalia_q(EW[k], det, reg) for k in chaves]; q = resume(mq)
            ver_q = "MELHOR" if q[0] >= 7 and q[3] <= 10 and q[6] <= 450 else "IGUAL" if q[0] >= 6 and 11 <= q[3] <= 14 and q[6] <= 700 else "PIOR"
            print(f"   Q t5 <- {kind:7s} H={H:<3d} ({len(mq)} variantes): detecção {q[0]:.0f}/7 ({q[1]:.0f}-{q[2]:.0f}) | FP {q[3]:.0f} ({q[4]:.0f}-{q[5]:.0f}) | fora {q[6]:.0f} h | totais {q[7]:.0f} h -> {ver_q}", flush=True)


if __name__ == "__main__":
    main()

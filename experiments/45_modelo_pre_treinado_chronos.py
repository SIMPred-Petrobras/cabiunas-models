#!/usr/bin/env python3
"""EXP 45 — UM MODELO DE SÉRIE TEMPORAL PRÉ-TREINADO (CHRONOS-BOLT, zero-shot) COMO DISCRIMINADOR DE FP E ACERTO NOS EPISÓDIOS DO v2.

PRÉ-REGISTRO (17/10/2026), COMMITADO ANTES DE RODAR.

POR QUE. A literatura recente (ICLR 2026) usa modelos de série temporal pré-treinados, sem ajuste, com calibração conforme; é a classe de modelo que eu não tinha testado. Seis tentativas anteriores de separar FP de acerto no disparo (EXP 36 a 40) falharam com outras informações. Aqui o modelo pré-treinado é uma terceira fonte: o erro de previsão zero-shot de cada sensor, que dispensa o treino no nosso dado.
O QUE SE FAZ (8 eventos; os episódios do detector publicado, sinais de produção): para cada episódio com referência causal (os 16 que nascem a partir de 01/03/2025: 9 FP e 7 de trip; os 2 de fev/2025 não têm os 60 dias de referência e ficam fora), as 26 tags de t e p a 10 min (uma de cada 5 amostras da grade de 2 min), contexto de 256 passos (42,7 h) terminando em ORIGEM = nascimento - 2 h, horizonte de 12 passos (2 h, até o nascimento: só passado); `amazon/chronos-bolt-tiny` e `amazon/chronos-bolt-small`, previsão mediana, NaN do contexto preenchido para frente e para trás. ERRO por tag = a média, no horizonte, de |observado - mediana prevista| dividida pela escala robusta da tag (MAD das diferenças de 10 min nos instantes vigiados dos 60 dias anteriores). REFERÊNCIA: o mesmo erro em 100 origens aleatórias dos instantes vigiados dos 60 dias ANTERIORES ao nascimento (causal, sem rótulo); z_tag = erro / percentil 95 da referência da própria tag; ESCORE do episódio = max z_tag (primário: as 26 tags; secundários: só as 14 de t e só as 12 de p).
MEDIDAS: AUC (trip > FP) do escore e p exato por permutação dos rótulos (todas as C(16,7) = 11.440 atribuições); confirmação por regra fixa (escore >= 1,0, o percentil 95 da própria referência): quantos FP e quantos trips ficam e a detecção e os FP depois de descartar os episódios não confirmados (os 2 de fev/2025 mantidos).
CRITÉRIOS (fixados antes). ÚTIL = no primário (bolt-small, as 26 tags) AUC >= 0,75, p da permutação <= 0,05 e, com a regra, detecção >= 7/8 e FP <= 6. Os secundários e o bolt-tiny são descritivos (correção: 6 testes, p <= 0,0083 para chamar de separação).
EXPECTATIVAS: AUC do primário entre 0,35 e 0,65 ~70%; ÚTIL ~4%; o bolt-small e o bolt-tiny com AUC a menos de 0,15 de diferença ~70%; a confirmação pela regra confirma >= 50% dos FP e dos trips ~55%.
INTERPRETAÇÃO PREVISTA. Se o modelo pré-treinado não separa FP de acerto, a quarta fonte independente (padrão de canais, alarmes da planta, modelos de normalidade, modelo pré-treinado) concorda: FP e acerto são indistinguíveis no disparo com a informação dos 38 tags; a lacuna é de informação sobre o desfecho. Se separar (AUC >= 0,75 com p <= 0,05), é a primeira informação nova e vira candidata a confirmação, a testar na sombra.
LIMITES. 16 episódios (9 FP e 7 trips): o poder é baixo (um AUC de 0,75 com 9 e 7 tem p ~0,06); o modelo é univariado (cada sensor sozinho); a escala de 10 min e o horizonte de 2 h são escolhas fixas; nada se adota sem sombra.

Uso:  PYTHONPATH=src .venv/bin/python experiments/45_modelo_pre_treinado_chronos.py
"""
from __future__ import annotations
import contextlib, io, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
from nv import io as nvio
from nv.causal import carrega_registro
from nv.avalia import avalia, carrega_controle, episodios_de_serie
from nv.controle_sinais import _ci, PKG

ROOT = Path(__file__).resolve().parents[1] / "data_local"; INICIO8 = pd.Timestamp("2025-02-01", tz="UTC"); JAN = pd.Timedelta(hours=48)
L = 256; H = 12; PASSO = 5


def auc(a, b):  # P(a > b)
    a, b = np.asarray(a, float), np.asarray(b, float); return float(((a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()) / (len(a) * len(b)))


def main():
    from chronos import BaseChronosPipeline
    g = nvio.carrega_grade(); ci = _ci()
    with contextlib.redirect_stdout(io.StringIO()):
        modelos = ci.carregar_modelos(PKG / "modelos"); det = modelos[-1]["detector"]
    tt = list(modelos[-1]["normalizacao"]["temperatura"]["cols"]); tp = list(modelos[-1]["normalizacao"]["pressao"]["cols"]); tags = tt + tp
    reg = [t for t in carrega_registro() if t >= pd.Timestamp("2025-02-27", tz="UTC")]; al_c, vig_c = carrega_controle("atual")
    S = pd.read_parquet(ROOT / "sinais_v2_8ev.parquet"); proc = S[["t", "p", "sp", "vb", "mask", "reset"]].copy(); proc["mask"] = proc["mask"].astype(bool); idx = proc.index
    with contextlib.redirect_stdout(io.StringIO()):
        R = ci.prever({"detector": det}, proc)
    al = R["is_anomaly"].astype(bool); eps = episodios_de_serie(al.loc[INICIO8:]); tipo = np.array([("trip" if any(a <= t and b >= t - JAN for t in reg) else "FP") for a, b in eps]); vig = vig_c.reindex(al_c.index).fillna(False)
    pub = avalia(al.reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8)
    X = g[tags].reindex(idx).iloc[::PASSO]; Xn = X.ffill().bfill().to_numpy(np.float32); ix = X.index; vm = vig_c.reindex(ix).fillna(False).to_numpy(bool); n_t = len(tt)
    escala = np.nanmedian(np.abs(np.diff(Xn, axis=0)), axis=0) * 1.4826 + 1e-6; rng = np.random.default_rng(20261016)
    validos = [i for i, (a, b) in enumerate(eps) if a >= pd.Timestamp("2025-03-01", tz="UTC")]; print(f"PUBLICADO: {pub.det}/8, FP {pub.fp}; episódios com referência: {len(validos)} ({int((tipo[validos] == 'FP').sum())} FP, {int((tipo[validos] == 'trip').sum())} trips)", flush=True)

    def erro(pipe, origens):
        """erro normalizado por tag para cada origem (índice no vetor de 10 min)."""
        ctx = torch.as_tensor(np.stack([Xn[o - L + 1:o + 1, :].T for o in origens]).reshape(-1, L)); out = []
        for a in range(0, len(ctx), 1024):
            q, _ = pipe.predict_quantiles(inputs=ctx[a:a + 1024], prediction_length=H, quantile_levels=[0.5]); out.append(q[:, :, 0].numpy())
        prev = np.concatenate(out).reshape(len(origens), len(tags), H); obs = np.stack([Xn[o + 1:o + 1 + H, :].T for o in origens]); return (np.abs(obs - prev).mean(axis=2) / escala[None, :])
    for nome in ("amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small"):
        pipe = BaseChronosPipeline.from_pretrained(nome, device_map="cpu", torch_dtype=torch.float32); sc_all, sc_t, sc_p = [], [], []
        for i in validos:
            a, b = eps[i]; o0 = ix.searchsorted(a - pd.Timedelta(hours=2)) - 1; ref_pos = np.flatnonzero(vm & (ix >= a - pd.Timedelta(days=60)) & (ix < a - pd.Timedelta(hours=6)) & (np.arange(len(ix)) > L + 5)); ref_pos = ref_pos[ref_pos + H < len(ix)]
            orr = rng.choice(ref_pos, size=min(100, len(ref_pos)), replace=False); e_ref = erro(pipe, list(orr)); p95 = np.quantile(e_ref, 0.95, axis=0) + 1e-9; e_ep = erro(pipe, [o0])[0] / p95
            sc_all.append(float(e_ep.max())); sc_t.append(float(e_ep[:n_t].max())); sc_p.append(float(e_ep[n_t:].max()))
        T = tipo[validos]; fp = np.flatnonzero(T == "FP"); tr = np.flatnonzero(T == "trip"); print(f"\n{nome}", flush=True)
        for rot, sc in (("26 tags (primário)", sc_all), ("14 tags de t", sc_t), ("12 tags de p", sc_p)):
            sc = np.array(sc); a0 = auc(sc[tr], sc[fp]); perm = []
            for comb in itertools.combinations(range(len(sc)), len(tr)):
                m = np.zeros(len(sc), bool); m[list(comb)] = True; perm.append(auc(sc[m], sc[~m]))
            p = float(np.mean(np.array(perm) >= a0 - 1e-12)); conf = sc >= 1.0
            al2 = al.copy()
            for j, i in enumerate(validos):
                if not conf[j]:
                    al2.loc[eps[i][0]:eps[i][1]] = False
            r = avalia(al2.reindex(al_c.index).fillna(False), vig, reg, inicio_periodo=INICIO8)
            print(f"   {rot:20s}: AUC(trip > FP) = {a0:.2f} (p exato da permutação = {p:.3f}) | confirma {int((conf[fp]).sum())} de {len(fp)} FP e {int((conf[tr]).sum())} de {len(tr)} trips | depois da regra: {r.det}/8, FP {r.fp}", flush=True)
        print(f"   escore (26 tags) nos FP: {[round(float(sc_all[j]), 2) for j in fp]} | nos trips: {[round(float(sc_all[j]), 2) for j in tr]}", flush=True)


if __name__ == "__main__":
    main()
